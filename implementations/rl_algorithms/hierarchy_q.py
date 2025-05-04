from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import random
import os
import logging

from implementations.core.torch.qformers import Model
from .base import Hierarchy_Base

import torch
import torch.nn as nn
from torch import optim

torch.autograd.set_detect_anomaly(False)


class Hierarchy_Q(Hierarchy_Base):

    def __init__(self, input_size, hidden_size, device, discount_factor=0.99, learning_rate=0.0001, entropy_weight=0.1, train_temperature=1.0) -> None:
        model = Model(input_size=input_size, hidden_size=hidden_size, device=device)
        optimizer = optim.Adam(model.parameters(), learning_rate)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5) 
        scheduler = None
        self.entropy_weight = entropy_weight
        super().__init__(model=model, optimizer=optimizer, scheduler=scheduler, device=device, discount_factor=discount_factor, train_temperature=train_temperature)


    def train(self, train_last_node, pivot: List[Any], train_data: List[Any], objective_tensor:Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        # pivot is a list of tuples (you will get this reward, moving from this context index, with the following available actions), must be sorted
        # train_data is a list of tuples (you use this action, to go from this pivot, to this pivot)

        context_size = state_tensor.size(0)
        action_size = action_list_tensor.size(1)
        num_pivot = len(pivot)

        # ----------------------
        # first prepare pivots

        # compute values of all context steps
        objective_tensor = torch.reshape(objective_tensor, [1, -1])
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])

        context_marks = torch.tensor([m for _, m, _ in pivot], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([r for r, _, _ in pivot], dtype=torch.float32, device=self.device)
        
        # now prepare available actions
        max_available_actions = max([len(aa) for _, _, aa in pivot])
        available_actions_indices = []
        action_set = set(action_list)
        for _, _, aa in pivot:
            # compute free actions
            free_actions = action_set - set(aa)
            new_aa = aa.copy()
            # now add missing
            while len(new_aa) < max_available_actions:
                new_aa.append(random.choice(list(free_actions)))
            available_actions_indices.append([action_list.index(a) for a in new_aa])
        available_actions_indices = torch.tensor(available_actions_indices, dtype=torch.int64, device=self.device) # shape: (num_pivot, max_action_length)
        
        # action_list_tensor has shape (all_action_length, action_size) must be expanded to (num_pivot, all_action_length, action_size)
        # available_actions_indices has shape (num_pivot, max_action_length) must be expanded to (num_pivot, max_action_length, action_size)
        available_actions_by_context = torch.gather(action_list_tensor.unsqueeze(0).expand(num_pivot, -1, -1), 1, available_actions_indices.unsqueeze(2).expand(-1, -1, action_size)) # shape: (num_pivot, max_action_length, action_size)
        available_actions_by_context = torch.reshape(available_actions_by_context, [1, num_pivot, max_available_actions, action_size])

        # self.model.train()
        # for rl using the model in eval mode to deactivate the normalization
        self.model.eval()
        action_scores, values = self.model(objective_tensor, state_tensor, available_actions_by_context, torch.reshape(context_marks, (1, -1)))
        action_scores = action_scores[0, :, :] # shape: (num_pivot, max_available_actions)
        state_values = values[0, :] # shape: (num_pivot)

        # ----------------------
        # now map to training data items
        train_action_indexes = torch.reshape(torch.tensor([pivot[p][2].index(a) for a, p, _ in train_data], dtype=torch.int64, device=self.device), (-1, 1))
        train_from_indexes = torch.tensor([p for _, p, _ in train_data], dtype=torch.int64, device=self.device)
        train_to_indexes = torch.tensor([p for _, _, p in train_data], dtype=torch.int64, device=self.device)
        train_action_scores = torch.gather(action_scores, 0, train_from_indexes.unsqueeze(-1).expand(-1, max_available_actions))

        train_state_values = torch.gather(state_values, 0, train_from_indexes)

        # now specialize truncated end
        if not train_last_node:
            last_value = state_values[-1].item()
            rewards = rewards[:-1]
        else:
            last_value = 0
            state_values = torch.concat([state_values, torch.zeros(1, device=self.device)], dim=0)

        with torch.no_grad():
            #  compute returns = rewards + gamma * next_state_q, but for all from to pivot
            # all_returns_2 = self._compute_snake_ladder_2(rewards, state_values)
            # train_returns_2 = all_returns_2[train_from_indexes, train_to_indexes]

            all_returns = self._compute_snake_ladder(rewards, last_value)
            train_returns = all_returns[train_from_indexes, train_to_indexes]

        # use vector instead of loops
        probs = torch.nn.functional.softmax(train_action_scores, dim=1)
        log_probs = torch.log(probs)
        current_scores = torch.gather(train_action_scores, 1, train_action_indexes)
        current_scores = current_scores.flatten()
        q_loss = (.5 * (current_scores - train_returns) ** 2.).sum()
        entropy = (-probs * log_probs).sum(dim=1).sum()
        loss = q_loss - self.entropy_weight * entropy
        is_nan = torch.isnan(loss)
        if is_nan:
            is_q_loss_nan = torch.isnan(q_loss).item()
            is_entropy_nan = torch.isnan(entropy).item()
            if not is_q_loss_nan:
                loss = q_loss
            else:
                logging.warning(f"Skipping nan: q loss nan {is_q_loss_nan}, entropy nan {is_entropy_nan}")
                return

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.iteration += 1

        self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()
