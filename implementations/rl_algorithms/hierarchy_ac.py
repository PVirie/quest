from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np
import random
import os
import logging

from implementations.core.torch.transformers import Command_Scorer

import torch
import torch.nn as nn
from torch import optim

torch.autograd.set_detect_anomaly(True)


class Value_Action:
    def __init__(self, state_value, selected_action, iteration=0):
        self.state_value = state_value
        self.selected_action = selected_action
        self.mdp_score = None
        self.available_actions = None

        self.has_released = False
        self.iteration = iteration

    def release(self):
        self.has_released = True


class Hierarchy_AC:
    LOG_ALPHA=0.95
    GAMMA = 0.97
    MAX_CONTEXT_SIZE = 128

    def __init__(self, input_size, device) -> None:
        self.device = device
        self.model = Command_Scorer(input_size=input_size, hidden_size=256, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.0001)

        self.ave_loss = 0
        self.iteration = 0

        # use mesh to build gammas
        x = torch.arange(0, self.MAX_CONTEXT_SIZE, dtype=torch.float32, device=device)
        y = torch.arange(0, -self.MAX_CONTEXT_SIZE, -1, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_xy = grid_x + grid_y
        self.gammas = torch.pow(self.GAMMA, grid_xy).to(device)
        # now only keep top right triangle
        self.gammas = torch.triu(self.gammas, diagonal=0)


    def save(self, dir_path):
        torch.save(self.model.state_dict(), os.path.join(dir_path, "model.pth"))
        torch.save(self.optimizer.state_dict(), os.path.join(dir_path, "optimizer.pth"))
        torch.save({
            'iteration': self.iteration,
            'ave_loss': self.ave_loss,
        }, os.path.join(dir_path, "state.pth"))


    def load(self, dir_path):
        if not os.path.exists(os.path.join(dir_path, "model.pth")):
            return False
        self.model.load_state_dict(torch.load(os.path.join(dir_path, "model.pth"), map_location=self.device))
        self.optimizer.load_state_dict(torch.load(os.path.join(dir_path, "optimizer.pth"), map_location=self.device))

        state = torch.load(os.path.join(dir_path, "state.pth"))
        self.iteration = state['iteration']
        self.ave_loss = state['ave_loss']
        return True


    def act(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True) -> Optional[str]:
        # action_list_tensor has shape (all_action_length, action_size)
        
        with torch.no_grad():
            objective_tensor = torch.reshape(objective_tensor, [1, -1])
            state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
            action_list_tensor = torch.reshape(action_list_tensor, [1, 1, -1, action_list_tensor.size(1)])

            # Get our next action and value prediction; only need the last state
            values, carry = self.model.evaluate_state(objective_tensor, state_tensor)
            action_scores = self.model.evaluate_actions(carry[:, -1:, :], action_list_tensor)[0, -1, :]
            values = values[0, -1, :].item()

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


    def _compute_snake_ladder(self, last_value, rewards):
        # this is not true snake ladder, but a simplified version
        # return a 2D matrix that contains best return from step i to step j
        # rewards is a 1D tensor
        # last_value is a scalar

        # use vector instead of loops
        context_length = rewards.size(0)
        last_value = last_value if isinstance(last_value, torch.Tensor) else torch.ones(1, device=self.device) * last_value
        
        # append last value to rewards
        R = torch.cat((rewards, last_value), dim=0)
        # gammas is [[r^0, r^1, ..., r^n], [0, r^0, r^1, ..., r^n-1], ...]
        # S_i = R_i + gamma*(S_i+1)
        S = torch.matmul(self.gammas[:context_length, :context_length + 1], R)
        
        # cumulative sum of R = [R[0], R[0] + R[1], R[0] + R[1] + R[2], ...]
        cR = torch.cumsum(R, dim=0)
        # now create from to sum table of R; i.e. K[i, j] = R[i] + R[i+1] + ... + R[j - 2] = cR[j - 2] - cR[i - 1]
        # use grid difference of cR
        grid_i, grid_j = torch.meshgrid(torch.arange(0, context_length, device=self.device), torch.arange(0, context_length + 1, device=self.device), indexing='ij')
        grid_ji = grid_j - grid_i
        K = torch.where(grid_ji > 0, cR[grid_j - 2] - cR[grid_i - 1], 0)

        # now we make final matrix W[i, j] = K[i, j] + S[j - 1]
        W = K + torch.where(grid_ji > 0, S[grid_j - 1], 0)

        return W


    def train(self, train_last_node, pivot: List[Any], train_data: List[Any], objective_tensor:Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        # pivot is a list of tuples (you will get this reward, moving from this context index, with the following available actions), must be sorted
        # train_data is a list of tuples (you use this action, to go from this pivot, to this pivot)

        context_size = state_tensor.size(0)
        action_size = action_list_tensor.size(1)

        # first compute values of all context steps
        objective_tensor = torch.reshape(objective_tensor, [1, -1])
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        values, carry = self.model.evaluate_state(objective_tensor, state_tensor)

        # now specialize truncated end
        if not train_last_node:
            pivot = pivot[:-1]
            last_value = values[0, -1, 0].item()
        else:
            last_value = 0

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
        carry = torch.gather(carry[0, :, :], 0, from_marks.unsqueeze(1).expand(-1, carry.size(2))) # shape: (train_data_length, carry_size)
        action_scores = self.model.evaluate_actions(torch.reshape(carry, (1, len(train_data), -1)), available_actions_by_context)
        action_scores = action_scores[0, :, :]
        values = torch.gather(values[0, :, 0], 0, from_marks)

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
        loss = policy_loss + 0.5 * value_loss - 1.0 * entropy # for many action, 1.0 seem to be optimal. (Originally it was 0.1.)
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


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        logging.info(msg)
        