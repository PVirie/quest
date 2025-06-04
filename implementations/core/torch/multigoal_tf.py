import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Multilayer_Relu, apply_transformer, causal_mask, positional_encoding


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Model, self).__init__()

        self.device = device
        self.hidden_size = hidden_size

        torch.manual_seed(42)  # For reproducibility
        self.embedding    = nn.Embedding(input_size, hidden_size, device=device)
        # state encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, device=device)
        self.context_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, device=device)
        self.objective_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, device=device)
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.value_decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.policy_decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)

        self.critic = Multilayer_Relu(hidden_size, 1, hidden_size, 1, device=device)

        self.pe = positional_encoding(1024, hidden_size).to(device) # 1024 is the maximum length of the context


    def forward(self, objectives, observations, actions, pivot_positions):
        # objectives has shape batch x objective_context_size
        # observations has shape batch x n_contexts x context_size
        # actions has shape batch x n_pivots x n_actions x action_size
        # pivot_positions has shape batch x n_pivots
        batch = objectives.size(0)
        objective_context_size = objectives.size(1)
        n_contexts = observations.size(1)
        context_size = observations.size(2)
        n_pivots = pivot_positions.size(1)
        n_actions = actions.size(2)
        action_size = actions.size(3)

        objective_embedding = self.embedding(objectives) # batch x objective_context_size x hidden
        objective_embedding = objective_embedding + self.pe[:objective_context_size, :] # add positional encoding
        objective_embedding = apply_transformer(self.objective_decoder, objective_embedding)
        
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = obs_embedding + self.pe[:context_size, :] # add positional encoding
        obs_embedding = apply_transformer(self.context_decoder, torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)))
        obs_embedding = torch.reshape(obs_embedding[:, 0, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = obs_embedding + self.pe[:n_contexts, :] # add positional encoding
        policy_internal = apply_transformer(self.policy_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        pivot_policy_internal = torch.gather(policy_internal, 1, pivot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)) # batch x n_pivots x hidden
        
        action_embedding = self.embedding(actions) # batch x n_pivots x n_actions x action_size x hidden
        action_embedding = action_embedding + self.pe[:action_size, :] # add positional encoding
        action_embedding = apply_transformer(self.action_decoder, torch.reshape(action_embedding, (-1, action_size, self.hidden_size)))
        action_embedding = torch.reshape(action_embedding[:, 0, :], (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden

        # now cross product between the state_internal and the action_embedding
        pre_actions = torch.reshape(pivot_policy_internal, (batch, n_pivots, self.hidden_size, 1)) # batch x n_pivots x hidden x 1
        scores = torch.matmul(action_embedding, pre_actions) # batch x n_pivots x n_actions x 1

        value_internal = apply_transformer(self.value_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        pivot_value_internal = torch.gather(value_internal, 1, pivot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)) # batch x n_pivots x hidden
        state_values = self.critic(pivot_value_internal) # batch x n_pivots x 1

        # return scores has shape batch x n_pivots x n_actions; is the scores of individual actions along the context length
        # state_values has shape batch x n_pivots; is the values of the best action from the current state
        return scores.squeeze(-1), state_values.squeeze(-1)
    