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
        self.optimizer = optim.Adam(self.model.parameters(), 0.0001)


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

        # first compute values of all context steps
        objective_tensor = torch.reshape(objective_tensor, [1, -1])
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])

        # compute all indices
        action_indexes = torch.reshape(torch.tensor([pivot[p][2].index(a) for a, p, _ in train_data], dtype=torch.int64, device=self.device), (-1, 1))
        from_indexes = torch.tensor([p for _, p, _ in train_data], dtype=torch.int64, device=self.device)
        to_indexes = torch.tensor([p for _, _, p in train_data], dtype=torch.int64, device=self.device)

        context_marks = torch.tensor([m for _, m, _ in pivot], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([r for r, _, _ in pivot], dtype=torch.float32, device=self.device)
        
        from_marks = torch.gather(context_marks, 0, from_indexes)

        # now prepare available actions
        max_available_actions = max([len(pivot[p][2]) for _, p, _ in train_data])
        available_actions_indices = []
        action_set = set(action_list)
        for _, p, _ in train_data:
            __, __, aa = pivot[p]
            # compute free actions
            free_actions = action_set - set(aa)
            new_aa = aa.copy()
            # now add missing
            while len(new_aa) < max_available_actions:
                new_aa.append(random.choice(list(free_actions)))
            available_actions_indices.append([action_list.index(a) for a in new_aa])
        available_actions_indices = torch.tensor(available_actions_indices, dtype=torch.int64, device=self.device) # shape: (train_data_length, action_length)
        
        # action_list_tensor has shape (all_action_length, action_size) must be expanded to (train_data_length, all_action_length, action_size)
        # available_actions_indices has shape (train_data_length, action_length) must be expanded to (train_data_length, action_length, action_size)
        available_actions_by_context = torch.gather(action_list_tensor.unsqueeze(0).expand(len(train_data), -1, -1), 1, available_actions_indices.unsqueeze(2).expand(-1, -1, action_size)) # shape: (train_data_length, action_length, action_size)
        
        available_actions_by_context = torch.reshape(available_actions_by_context, [1, len(train_data), -1, action_size])
        action_scores, values = self.model(objective_tensor, state_tensor, available_actions_by_context, torch.reshape(from_marks, (1, -1)))
        action_scores = action_scores[0, :, :]
        values = values[0, :]

        # now specialize truncated end
        if not train_last_node:
            last_value = values[-1].item()
            rewards = rewards[:-1]
        else:
            last_value = 0

        with torch.no_grad():
            all_returns = self._compute_snake_ladder(last_value, rewards)
            returns = all_returns[from_indexes, to_indexes]
            advantages = returns - values

        # use vector instead of loops
        probs = torch.nn.functional.softmax(action_scores, dim=1)
        log_probs = torch.log(probs)
        log_action_probs = torch.clamp(torch.gather(log_probs, 1, action_indexes), min=-8)
        log_action_probs = log_action_probs.flatten()
        policy_loss = (-log_action_probs * advantages).mean()
        value_loss = (.5 * (values - returns) ** 2.).mean()
        entropy = (-probs * log_probs).sum(dim=1).mean()
        loss = policy_loss + 0.5 * value_loss - 2.0 * entropy # for many action, 1.0 seem to be optimal. (Originally it was 0.1.)
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
