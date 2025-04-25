from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import random
import os
import logging

from implementations.core.torch.transformers import Model
from implementations.core.torch.base import softmax_with_temperature
from .base import Value_Action, Hierarchy_Base

import torch
import torch.nn as nn
from torch import optim

torch.autograd.set_detect_anomaly(False)



class Hierarchy_AC(Hierarchy_Base):

    def __init__(self, input_size, device) -> None:
        super().__init__(device)
        self.model = Model(input_size=input_size, hidden_size=128, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00001)


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

            action_scores, _ = self.model(objective_tensor, state_tensor, action_list_tensor, pivot_positions)
            action_scores = action_scores[0, 0, :]

            if sample_action:
                # sample
                self.model.train()
                probs = softmax_with_temperature(action_scores, temperature=2.0, dim=0)  # n_actions
                index = torch.multinomial(probs, num_samples=1).item() # 1
                rank = torch.argsort(action_scores, descending=True).tolist().index(index) + 1
            else:
                # greedy
                self.model.eval()
                index = torch.argmax(action_scores, dim=0).item()
                rank = 1

        return Value_Action(action_list[index], rank, self.iteration)


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
        
        self.model.train()
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

        with torch.no_grad():
            all_returns = self._compute_snake_ladder(rewards, last_value)
            train_returns = all_returns[train_from_indexes, train_to_indexes]
            train_advantages = train_returns - train_state_values

        # use vector instead of loops
        probs = torch.nn.functional.softmax(train_action_scores, dim=1)
        log_probs = torch.log(probs)
        log_action_probs = torch.clamp(torch.gather(log_probs, 1, train_action_indexes), min=-8)
        log_action_probs = log_action_probs.flatten()
        policy_loss = (-log_action_probs * train_advantages).mean()
        value_loss = (.5 * (train_state_values - train_returns) ** 2.).mean()
        entropy = (-probs * log_probs).sum(dim=1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.02 * entropy # entropy has to be adjusted, too low and it will get stuck at a command.
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
