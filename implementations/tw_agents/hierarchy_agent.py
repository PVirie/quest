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

        self.has_released = False
        self.iteration = iteration

    def release(self):
        self.has_released = True


class Hierarchy_Agent:
    LOG_ALPHA=0.95
    GAMMA = 0.95
    MAX_CONTEXT_SIZE = 32

    def __init__(self, input_size, device) -> None:
        self.device = device
        self.model = Command_Scorer(input_size=input_size, hidden_size=64, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

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


    def act(self, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True) -> Optional[str]:
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        action_scores, values = self.model(state_tensor, action_list_tensor)

        action_scores = action_scores[0, -1, :]
        values = values[0, -1, :].item()

        if sample_action:
            probs = torch.nn.functional.softmax(action_scores, dim=0)  # n_actions
            indices = torch.multinomial(probs, num_samples=1).item() # 1
        else:
            # greedy
            indices = torch.argmax(action_scores, dim=0).item()

        return Value_Action(values, action_list[indices], self.iteration)


    def _discount_rewards(self, last_value, values, rewards):
        # returns, advantages = [], []
        # R = last_values.item() if isinstance(last_values, torch.Tensor) else last_values
        # for i in range(len(rewards) - 1, -1, -1):
        #     R = rewards[i] + self.GAMMA * R
        #     adv = R - values[i].item()
        #     returns.append(R)
        #     advantages.append(adv)

        # return returns[::-1], advantages[::-1]
        # use vector instead of loops

        last_value = last_value if isinstance(last_value, torch.Tensor) else torch.ones(1, device=self.device) * last_value
        # append last value to rewards
        R = torch.cat((rewards, last_value), dim=0)
        context_length = R.size(0)
        # gammas is [[r^0, r^1, ..., r^n], [0, r^0, r^1, ..., r^n-1], ...]
        returns = torch.matmul(self.gammas[:context_length - 1, :context_length], R)
        advantages = returns - values
        
        return returns.detach(), advantages.detach()


    def train(self, last_value, transitions: List[Any], state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        # transitions is a list of tuples (you will get this reward, if you select this action, from this context)
        
        selected_action_set = set([action for _, action, _ in transitions])
        unused_actions = set(action_list) - selected_action_set
        # make a new list, fill the rest with unused actions
        action_list = list(selected_action_set) + random.sample(list(unused_actions), min(10, len(unused_actions)))

        context_marks = torch.tensor([m for _, _, m in transitions], dtype=torch.int64, device=self.device)

        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        action_scores, values = self.model(state_tensor, action_list_tensor)

        action_scores = torch.gather(action_scores[0, :, :], 0, torch.unsqueeze(context_marks, 1).expand(-1, action_scores.size(2)))
        values = torch.gather(values[0, :, 0], 0, context_marks)

        action_indexes = torch.reshape(torch.tensor([action_list.index(a) for _, a, _ in transitions], dtype=torch.int64, device=self.device), (-1, 1))
        rewards = torch.tensor([r for r, _, _ in transitions], dtype=torch.float32, device=self.device)

        returns, advantages = self._discount_rewards(last_value, values.flatten().detach(), rewards)

        # loss = 0
        # for transition, ret, adv in zip(transitions, returns, advantages):
        #     reward_, tf = transition
        #     if tf.iteration != self.iteration:
        #         # skip
        #         continue
            
        #     probs            = F.softmax(tf.action_scores, dim=0)
        #     log_probs        = torch.log(probs)
        #     log_action_probs = log_probs[tf.indexes]
        #     policy_loss      = (-log_action_probs * adv).sum()
        #     value_loss       = (.5 * (tf.values - ret) ** 2.).sum()
        #     entropy          = (-probs * log_probs).sum()
        #     loss            += policy_loss + 0.5 * value_loss - 0.1 * entropy

        # use vector instead of loops
        probs = torch.nn.functional.softmax(action_scores, dim=1)
        log_probs = torch.log(probs)
        log_action_probs = torch.gather(log_probs, 1, action_indexes)
        log_action_probs = torch.clip(log_action_probs.flatten(), min=-8)
        policy_loss = (-log_action_probs * advantages).sum()
        value_loss = (.5 * (values - returns) ** 2.).sum()
        entropy = (-probs * log_probs).sum()
        loss = policy_loss + 0.5 * value_loss - 0.1 * entropy

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
        