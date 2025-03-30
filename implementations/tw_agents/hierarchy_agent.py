from typing import List, Mapping, Any, Optional
from collections import defaultdict
import numpy as np

from implementations.core.torch.transformers import Command_Scorer

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class Hierarchy_Agent:
    UPDATE_FREQUENCY = 10
    LOG_FREQUENCY = 1000
    LOG_ALPHA=0.99
    GAMMA = 0.9

    def __init__(self, input_size, device) -> None:
        self._initialized = False
        self._epsiode_has_started = False

        self.model = Command_Scorer(input_size=input_size, hidden_size=128, device=device)
        self.optimizer = optim.Adam(self.model.parameters(), 0.00003)

        self.mode = "test"

    def train(self):
        self.mode = "train"
        self.no_train_step = 1
        self.ave_loss = 0

    def test(self):
        self.mode = "test"


    def _discount_rewards(self, last_values, transitions):
        returns, advantages = [], []
        R = last_values.item()
        last_score = 0
        rewards = []
        for t in transitions:
            rewards.append(t[0] - last_score)
            last_score = t[0]
        for t, r in zip(reversed(transitions), reversed(rewards)):
            _, _, _, values = t
            R = r + self.GAMMA * R
            adv = (R - values).detach()
            returns.append(R)
            advantages.append(adv)

        return returns[::-1], advantages[::-1]

    def act(self, state_tensor: Any, action_list_tensor: Any, transitions: List[Any], infos: Mapping[str, Any]) -> Optional[str]:
        # transitions is a list of (score, indexes, outputs, values)
        state_tensor = torch.reshape(state_tensor, [1, -1, state_tensor.size(1)])
        action_list_tensor = torch.reshape(action_list_tensor, [1, -1, action_list_tensor.size(1)])

        # Get our next action and value prediction.
        outputs, indexes, values = self.model(state_tensor, action_list_tensor)
        outputs = outputs[0, -1, :]
        indexes = indexes[0, -1, :].item()
        values = values[0, -1, :]
        action = infos["admissible_commands"][indexes]

        if self.mode == "train":
            self.no_train_step += 1

            if self.no_train_step % self.UPDATE_FREQUENCY == 0:
                # Update model
                returns, advantages = self._discount_rewards(values, transitions)

                loss = 0
                for transition, ret, advantage in zip(transitions, returns, advantages):
                    score_, indexes_, outputs_, values_ = transition

                    advantage        = advantage.detach() # Block gradients flow here.
                    probs            = F.softmax(outputs_, dim=0)
                    log_probs        = torch.log(probs)
                    log_action_probs = log_probs[indexes_]
                    policy_loss      = (-log_action_probs * advantage).sum()
                    value_loss       = (.5 * (values_ - ret) ** 2.).sum()
                    entropy     = (-probs * log_probs).sum()
                    loss += policy_loss + 0.5 * value_loss - 0.1 * entropy

                self.ave_loss = self.LOG_ALPHA * self.ave_loss + (1 - self.LOG_ALPHA) * loss.item()
                if self.no_train_step % self.LOG_FREQUENCY == 0:
                    msg = "{:6d}. ".format(self.no_train_step)
                    msg += "loss: {:5.2f}; ".format(self.ave_loss)
                    print(msg)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                self.optimizer.step()
                self.optimizer.zero_grad()

            return action, indexes, outputs, values
        else:
            return action, None, None, None