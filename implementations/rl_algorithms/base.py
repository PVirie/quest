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

torch.autograd.set_detect_anomaly(False)



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


class Hierarchy_Base:
    LOG_ALPHA=0.95
    GAMMA = 0.97
    MAX_CONTEXT_SIZE = 128

    def __init__(self, device):
        self.device = device

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
        torch.save({
            'iteration': self.iteration,
            'ave_loss': self.ave_loss,
        }, os.path.join(dir_path, "state.pth"))


    def load(self, dir_path):
        if not os.path.exists(os.path.join(dir_path, "state.pth")):
            return False
        state = torch.load(os.path.join(dir_path, "state.pth"))
        self.iteration = state['iteration']
        self.ave_loss = state['ave_loss']
        return True


    def print(self, step):
        msg = "{:6d}. ".format(step)
        msg += "loss: {:5.2f}; ".format(self.ave_loss)
        logging.info(msg)


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
    

    def act(self, objective_tensor: Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str], sample_action=True) -> Optional[str]:
        raise NotImplementedError("act() is not implemented in base class")


    def train(self, train_last_node, pivot: List[Any], train_data: List[Any], objective_tensor:Any, state_tensor: Any, action_list_tensor: Any, action_list: List[str]):
        raise NotImplementedError("train() is not implemented in base class")


        