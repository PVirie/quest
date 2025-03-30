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
    def __init__(self, indexes, outputs, values):
        self.indexes = indexes
        self.outputs = outputs
        self.values = values

        self.has_released = False

    def release(self):
        self.indexes = None
        self.outputs = None
        self.values = None

        self.has_released = True


class Hierarchy_Agent:
    LOG_ALPHA=0.99
    GAMMA = 0.9

    def __init__(self, input_size, device) -> None:
        self.model = Command_Scorer(input_size=input_size, hidden_size=128, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        self.ave_loss = 0


    def act(self, state_tensor: Any, action_list_tensor: Any, infos: Mapping[str, Any]) -> Optional[str]:
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(state_tensor, action_list_tensor)
        outputs = outputs[0, -1, :]
        indexes = indexes[0, -1, :].item()
        values = values[0, -1, :]
        action = infos["admissible_commands"][indexes]

        return action, Tensors_Ref(indexes, outputs, values)


    def _discount_rewards(self, last_values, transitions):
        # transitions is a list of (reward, Tensors_Ref(indexes, outputs, values))
        returns, advantages = [], []
        R = last_values.item() if isinstance(last_values, torch.Tensor) else last_values
        for t in reversed(transitions):
            r, tf = t
            R = r + self.GAMMA * R
            adv = R - tf.values.item()
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]


    def train(self, last_values, transitions: List[Any]):
        returns, advantages = self._discount_rewards(last_values, transitions)
        loss = 0
        for transition, ret, adv in zip(transitions, returns, advantages):
            reward_, tf = transition
            
            probs            = F.softmax(tf.outputs, dim=0)
            log_probs        = torch.log(probs)
            log_action_probs = log_probs[tf.indexes]
            policy_loss      = (-log_action_probs * adv).sum()
            value_loss       = (.5 * (tf.values - ret) ** 2.).sum()
            entropy          = (-probs * log_probs).sum()
            loss            += policy_loss + 0.5 * value_loss - 0.1 * entropy

        self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
        self.optimizer.zero_grad()

        for _, tf in transitions:
            tf.release()


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        print(msg)
        