from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import random
import os
import logging

from implementations.core.torch.qformers import Q_Table
from .base import Value_Action, Hierarchy_Base

import torch
import torch.nn as nn
from torch import optim

torch.autograd.set_detect_anomaly(False)


class Hierarchy_Q(Hierarchy_Base):

    def __init__(self, input_size, device) -> None:
        super().__init__(device)
        self.model = Q_Table(input_size=input_size, hidden_size=128, num_output_qs=16, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00005)


    def save(self, dir_path):
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dir_path, "optimizer.pth"))
        super().save(dir_path)


    def load(self, dir_path):
        result = super().load(dir_path)
        if not result:
            return result
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "model.pth"), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(dir_path, "optimizer.pth"), map_location=self.device))
        return True


    def act(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True) -> Optional[str]:
        # action_list_tensor has shape (all_action_length, action_size)
        n_context = state_tensor.size(0)
        
        with torch.no_grad():
            objective_tensor = torch.reshape(objective_tensor, [1, -1])
            state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
            action_list_tensor = torch.reshape(action_list_tensor, [1, 1, -1, action_list_tensor.size(1)])
            pivot_positions = torch.tensor([[n_context - 1]], dtype=torch.int64, device=self.device) # shape: (1, 1)

            action_scores, values = self.model(objective_tensor, state_tensor, action_list_tensor, pivot_positions)
            action_scores = action_scores[0, 0, :]
            values = values.item()

            if sample_action:
                # lower_bound = torch.min(action_scores)
                # sample_bias = lower_bound + 0.2 * (torch.max(action_scores) - lower_bound)
                # action_scores = torch.clip(action_scores, min=sample_bias) # further improve exploration
                probs = torch.nn.functional.softmax(action_scores, dim=0)  # n_actions
                indices = torch.multinomial(probs, num_samples=1).item() # 1
            else:
                # greedy
                indices = torch.argmax(action_scores, dim=0).item()

        return Value_Action(values, action_list[indices], self.iteration)


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
        action_scores, values = self.model(objective_tensor, state_tensor, available_actions_by_context, torch.reshape(context_marks, (1, -1)))
        action_scores = action_scores[0, :, :] # shape: (num_pivot, max_available_actions)
        state_q = values[0, :] # shape: (num_pivots)

        # now specialize truncated end
        if not train_last_node:
            rewards = rewards[:-1]
        else:
            state_q = torch.concat([state_q, 0], dim=0)

        # ----------------------
        # now map to training data items
        train_action_indexes = torch.reshape(torch.tensor([pivot[p][2].index(a) for a, p, _ in train_data], dtype=torch.int64, device=self.device), (-1, 1))
        train_from_indexes = torch.tensor([p for _, p, _ in train_data], dtype=torch.int64, device=self.device)
        train_to_indexes = torch.tensor([p for _, _, p in train_data], dtype=torch.int64, device=self.device)
        train_action_scores = torch.gather(action_scores, 0, train_from_indexes.unsqueeze(-1).expand(-1, max_available_actions))

        with torch.no_grad():
            #  compute returns = rewards + self.gammas * next_state_q, but for all from to pivot
            all_returns = self._compute_snake_ladder_2(rewards, state_q)
            train_returns = all_returns[train_from_indexes, train_to_indexes]

        # use vector instead of loops
        probs = torch.nn.functional.softmax(train_action_scores, dim=1)
        log_probs = torch.log(probs)
        current_scores = torch.gather(train_action_scores, 1, train_action_indexes)
        current_scores = current_scores.flatten()
        q_loss = (.5 * (current_scores - train_returns) ** 2.).mean()
        entropy = (-probs * log_probs).sum(dim=1).mean()
        loss = q_loss - 1.0 * entropy # for many action, 1.0 seem to be optimal. (Originally it was 0.1.)
        is_nan = torch.isnan(loss)
        if is_nan:
            logging.warning("Loss is NaN, skipping training")
            return

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.iteration += 1

        self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()
