import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Q_Table


def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    return pe


def causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask


class Command_Scorer(nn.Module, Q_Table):
    def __init__(self, input_size, hidden_size, device):
        super(Command_Scorer, self).__init__()

        self.device = device

        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.critic = nn.Linear(hidden_size, 1, device=device)
        self.actor = nn.Linear(hidden_size, hidden_size, device=device)

        self.pe = positional_encoding(512, hidden_size).to(device) # 512 is the maximum length of the context


    def forward(self, observations, actions, **kwargs):
        # observations has shape batch x n_contexts x input_size
        # actions has shape batch x n_actions x input_size
        n_contexts = observations.size(1)
        n_actions = actions.size(1)
        hidden_size = observations.size(2)

        obs_embedding = self.embedding(observations) # batch x n_contexts x hidden
        action_embedding = self.embedding(actions) # batch x n_actions x hidden

        obs_embedding = obs_embedding + self.pe[:n_contexts, :].to(self.device) # add positional encoding

        # accept (sequence_length, batch_size, d_model)
        obs_embedding = obs_embedding.permute(1, 0, 2)
        state_internal = self.transformer_decoder(obs_embedding, obs_embedding, tgt_mask=causal_mask(n_contexts).to(self.device), tgt_is_causal=True) # n_contexts x batch x hidden
        state_internal = state_internal.permute(1, 0, 2) # batch x n_contexts x hidden
        values = self.critic(state_internal)

        # now cross product between the state_internal and the action_embedding
        pre_actions = self.actor(state_internal) # batch x n_contexts x hidden
        scores = torch.matmul(pre_actions, action_embedding.permute(0, 2, 1)) # batch x n_contexts x n_actions

        probs = F.softmax(scores, dim=2)  # batch x n_contexts x n_actions
        indices = torch.multinomial(torch.reshape(probs, (-1, n_actions)), num_samples=1)
        indices = torch.reshape(indices, (-1, n_contexts, 1))

        # scores has shape batch x n_contexts x n_actions; is the scores of individual actions along the context length
        # indices has shape batch x n_contexts x 1; is the indices of the actions chosen along the context length
        # values has shape batch x n_contexts x 1; is the values of the states
        return scores, indices, values


    def reset_hidden(self, batch_size):
        pass

