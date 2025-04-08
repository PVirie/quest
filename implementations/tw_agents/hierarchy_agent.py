from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np

from implementations.core.torch.transformers import Command_Scorer

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)


class Tensors_Ref:
    def __init__(self, selected_index, state_value, iteration=0):
        self.selected_index = selected_index
        self.selection_string = None

        self.state_value = state_value

        self.has_released = False
        self.iteration = iteration

    
    def change_selected_index(self, selected_index):
        self.selected_index = selected_index


    def set_selection_string(self, selection_string):
        self.selection_string = selection_string


    def release(self):
        self.has_released = True


class Hierarchy_Agent:
    LOG_ALPHA=0.99
    GAMMA = 0.9

    def __init__(self, input_size, device) -> None:
        self.device = device
        self.model = Command_Scorer(input_size=input_size, hidden_size=64, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        self.ave_loss = 0
        self.iteration = 0


    def act(self, state_tensor: Any, action_list_tensor: Any) -> Optional[str]:
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        indexes, action_scores, values, internal_states = self.model(state_tensor, action_list_tensor)
        indexes = indexes[0, -1, :].item()
        action_scores = action_scores[0, -1, :]
        values = values[0, -1, :].item()
        internal_states = internal_states[0, -1, :]

        return Tensors_Ref(indexes, values, self.iteration)


    def _discount_rewards(self, last_values, values, transitions):
        # transitions is a list of (reward, Tensors_Ref(indexes, outputs, values))
        returns, advantages = [], []
        R = last_values.item() if isinstance(last_values, torch.Tensor) else last_values
        for i in range(len(transitions) - 1, -1, -1):
            r, _, _ = transitions[i]
            R = r + self.GAMMA * R
            adv = R - values[i].item()
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]


    def train(self, last_values, transitions: List[Any], state_tensor: Any, action_list_tensor: Any):

        len_transitions = len(transitions)

        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        _, action_scores, values, internal_states = self.model(state_tensor, action_list_tensor)
        action_scores = action_scores[0, -len_transitions:, :]
        values = values[0, -len_transitions:, 0]

        indexes = torch.reshape(torch.tensor([index for _, index, _ in transitions], dtype=torch.int64, device=self.device), (-1, 1))

        returns, advantages = self._discount_rewards(last_values, values.flatten(), transitions)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

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
        probs = F.softmax(action_scores, dim=1)
        log_probs = torch.log(probs)
        log_action_probs = torch.gather(log_probs, 1, indexes)
        log_action_probs = log_action_probs.flatten()
        policy_loss = (-log_action_probs * advantages).sum()
        value_loss = (.5 * (values - returns) ** 2.).sum()
        entropy = (-probs * log_probs).sum()
        loss = policy_loss + 0.5 * value_loss - 0.1 * entropy


        if isinstance(loss, torch.Tensor):
            self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.iteration += 1

        for _, _, tf in transitions:
            if tf is not None and not tf.has_released:
                tf.release()


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        print(msg)
        