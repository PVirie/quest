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
    return pe


def causal_mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask


def apply_transformer(input, decoder, tgt_mask, tgt_is_causal=False):
    input = input.permute(1, 0, 2)
    output = decoder(input, input, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal) # n_contexts x batch x hidden
    output = output.permute(1, 0, 2) # batch x n_contexts x hidden
    return output


class Command_Scorer(nn.Module, Q_Table):
    def __init__(self, input_size, hidden_size, device):
        super(Command_Scorer, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, device=device)
        self.context_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, device=device)
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.state_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.critic = nn.Linear(hidden_size, 1, device=device)
        self.actor = nn.Linear(hidden_size, hidden_size, device=device)

        self.pe = positional_encoding(512, hidden_size).to(device) # 512 is the maximum length of the context


    def forward(self, observations, actions, **kwargs):
        # observations has shape batch x n_contexts x context_size
        # actions has shape batch x n_actions x action_size

        values, state_internal = self.evaluate_state(observations, **kwargs)
        scores = self.evaluate_actions(state_internal, actions, **kwargs)

        # scores has shape batch x n_contexts x n_actions; is the scores of individual actions along the context length
        # values has shape batch x n_contexts x 1; is the values of the states
        return scores, values


    def evaluate_state(self, observations, **kwargs):
        n_contexts = observations.size(1)
        context_size = observations.size(2)
    
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = obs_embedding + self.pe[:context_size, :] # add positional encoding
        obs_embedding = apply_transformer(torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)), self.context_decoder, tgt_mask=causal_mask(context_size, self.device), tgt_is_causal=True)
        obs_embedding = torch.reshape(obs_embedding[:, -1, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = obs_embedding + self.pe[:n_contexts, :] # add positional encoding
        state_internal = apply_transformer(obs_embedding, self.state_decoder, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        values = self.critic(state_internal)

        return values, state_internal


    def evaluate_actions(self, state_internal, actions, **kwargs):
        n_actions = actions.size(1)
        action_size = actions.size(2)

        action_embedding = self.embedding(actions) # batch x n_actions x action_size x hidden
        action_embedding = action_embedding + self.pe[:action_size, :] # add positional encoding
        action_embedding = apply_transformer(torch.reshape(action_embedding, (-1, action_size, self.hidden_size)), self.action_decoder, tgt_mask=causal_mask(action_size, self.device), tgt_is_causal=True) 
        action_embedding = torch.reshape(action_embedding[:, -1, :], (-1, n_actions, self.hidden_size)) # batch x n_actions x hidden

        # now cross product between the state_internal and the action_embedding
        pre_actions = self.actor(state_internal) # batch x n_contexts x hidden
        scores = torch.matmul(pre_actions, action_embedding.permute(0, 2, 1)) # batch x n_contexts x n_actions

        return scores