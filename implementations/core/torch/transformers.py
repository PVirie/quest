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
    """
    in pytorch 2.6, mask is a boolean tensor where True means to be masked out.
    """
    # make
    # tensor([
    #     [F, T, T, T],
    #     [F, F, T, T],
    #     [F, F, F, T],
    #     [F, F, F, F]
    # ])
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask


def apply_transformer(decoder, input, memory=None, tgt_mask=None, tgt_is_causal=False):
    input = input.permute(1, 0, 2)
    if memory is not None:
        memory = memory.permute(1, 0, 2)
    else:
        memory = input
    output = decoder(input, memory, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal) # n_contexts x batch x hidden
    output = output.permute(1, 0, 2) # batch x n_contexts x hidden
    return output


class Multilayer_Relu(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers=1, device=None):
        super(Multilayer_Relu, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size, device=device))
        self.layers.append(nn.Linear(hidden_size, output_size, device=device))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


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

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 1, device=device)
        self.actor = Multilayer_Relu(hidden_size, hidden_size, hidden_size, 1, device=device)

        self.pe = positional_encoding(256 + 128, hidden_size).to(device) # 256 is the maximum length of the context


    def forward(self, objectives, observations, actions, **kwargs):
        # objectives has shape batch x objective_context_size
        # observations has shape batch x n_contexts x context_size
        # actions has shape batch x n_actions x action_size

        values, state_internal = self.evaluate_state(objectives, observations, **kwargs)
        scores = self.evaluate_actions(state_internal, actions, **kwargs)

        # scores has shape batch x n_contexts x n_actions; is the scores of individual actions along the context length
        # values has shape batch x n_contexts x 1; is the values of the states
        return scores, values


    def evaluate_state(self, objectives, observations, **kwargs):
        objective_context_size = objectives.size(1)
        n_contexts = observations.size(1)
        context_size = observations.size(2)

        objective_embedding = self.embedding(objectives) # batch x objective_context_size x hidden
        objective_embedding = objective_embedding + self.pe[:objective_context_size, :] # add positional encoding
        
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = obs_embedding + self.pe[:context_size, :] # add positional encoding
        obs_embedding = apply_transformer(self.context_decoder, torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)))
        obs_embedding = torch.reshape(obs_embedding[:, 0, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = obs_embedding + self.pe[:n_contexts, :] # add positional encoding
        state_internal = apply_transformer(self.state_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        values = self.critic(state_internal)

        return values, state_internal


    def evaluate_actions(self, state_internal, actions, **kwargs):
        n_actions = actions.size(1)
        action_size = actions.size(2)

        action_embedding = self.embedding(actions) # batch x n_actions x action_size x hidden
        action_embedding = action_embedding + self.pe[:action_size, :] # add positional encoding
        action_embedding = apply_transformer(self.action_decoder, torch.reshape(action_embedding, (-1, action_size, self.hidden_size)))
        action_embedding = torch.reshape(action_embedding[:, 0, :], (-1, n_actions, self.hidden_size)) # batch x n_actions x hidden

        # now cross product between the state_internal and the action_embedding
        pre_actions = self.actor(state_internal) # batch x n_contexts x hidden
        scores = torch.matmul(pre_actions, action_embedding.permute(0, 2, 1)) # batch x n_contexts x n_actions

        return scores