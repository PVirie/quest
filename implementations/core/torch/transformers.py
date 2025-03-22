import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from .base import Q_Table


class Stacked_Decoder_Only_Transformers(nn.Module, Q_Table):
    def __init__(self, input_size, hidden_size, device):
        super(Stacked_Decoder_Only_Transformers, self).__init__()

        self.device = device

        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        self.critic       = nn.Linear(hidden_size, 1, device=device)
        self.att_cmd      = nn.Linear(hidden_size * 2, 1, device=device)


    def forward(self, observations, actions, **kwargs):
        pass