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
    def __init__(self, indexes, action_scores, values, internal_states, iteration=0):
        self.indexes = indexes
        self.action_scores = action_scores
        self.values = values
        self.internal_states = internal_states

        self.has_released = False
        self.iteration = iteration


    def override_selected_action(self, indexes):
        self.indexes = indexes


    def release(self):
        self.action_scores = self.action_scores.detach().clone()
        self.values = self.values.detach().clone()
        self.internal_states = self.internal_states.detach().clone()

        self.has_released = True


class Hierarchy_Agent:
    LOG_ALPHA=0.99
    GAMMA = 0.9

    def __init__(self, input_size, device) -> None:
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
        values = values[0, -1, :]
        internal_states = internal_states[0, -1, :]

        return Tensors_Ref(indexes, action_scores, values, internal_states, self.iteration)


    def append_action(self, tf: Tensors_Ref, new_action_list_tensor: Any) -> Optional[str]:
        # return the original size and the new size
        action_scores = self.model.evaluate_actions(tf.internal_states.unsqueeze(0), new_action_list_tensor.unsqueeze(0))
        action_scores = action_scores[0, -1, :]
        original_size = tf.action_scores.size(0)
        tf.action_scores = torch.concat([tf.action_scores, action_scores], dim=0)
        return original_size, tf.action_scores.size(0)


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
            if tf.iteration != self.iteration:
                # skip
                continue
            
            probs            = F.softmax(tf.action_scores, dim=0)
            log_probs        = torch.log(probs)
            log_action_probs = log_probs[tf.indexes]
            policy_loss      = (-log_action_probs * adv).sum()
            value_loss       = (.5 * (tf.values - ret) ** 2.).sum()
            entropy          = (-probs * log_probs).sum()
            loss            += policy_loss + 0.5 * value_loss - 0.1 * entropy

        if isinstance(loss, torch.Tensor):
            self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.iteration += 1

        for _, tf in transitions:
            tf.release()


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        print(msg)
        