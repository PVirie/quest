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

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.context_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.state_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=16, device=device)
        self.q_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.critic = Multilayer_Relu(hidden_size, hidden_size, hidden_size, 2, device=device)

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
        
        obs_embedding = self.embedding(observations) # batch x n_contexts x context_size x hidden
        obs_embedding = obs_embedding + self.pe[:context_size, :] # add positional encoding
        obs_embedding = apply_transformer(self.context_decoder, torch.reshape(obs_embedding, (-1, context_size, self.hidden_size)))
        obs_embedding = torch.reshape(obs_embedding[:, 0, :], (-1, n_contexts, self.hidden_size)) # batch x n_contexts x hidden

        obs_embedding = obs_embedding + self.pe[:n_contexts, :] # add positional encoding
        state_internal = apply_transformer(self.state_decoder, obs_embedding, memory=objective_embedding, tgt_mask=causal_mask(n_contexts, self.device), tgt_is_causal=True) # batch x n_contexts x hidden
        
        pivot_state_internal = torch.gather(state_internal, 1, pivot_positions.unsqueeze(-1).expand(-1, -1, self.hidden_size)) # batch x n_pivots x hidden

        action_embedding = self.embedding(actions) # batch x n_pivots x n_actions x action_size x hidden
        action_embedding = action_embedding + self.pe[:action_size, :] # add positional encoding
        action_embedding = apply_transformer(self.action_decoder, torch.reshape(action_embedding, (-1, action_size, self.hidden_size)))
        action_embedding = torch.reshape(action_embedding[:, 0, :], (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden

        collapsed_pivot_state_internal = torch.reshape(pivot_state_internal, (-1, 1, self.hidden_size))
        collapsed_actions = torch.reshape(action_embedding, (-1, n_actions, self.hidden_size))

        scores = apply_transformer(self.q_decoder, collapsed_actions, memory=collapsed_pivot_state_internal)
        scores = torch.reshape(scores, (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden
        scores = scores[:, :, :, 0] # batch x n_pivots x n_actions

        pre_qs = torch.reshape(self.critic(pivot_state_internal), (batch, n_pivots, self.hidden_size, 1)) # batch x n_pivots x hidden x 1
        qs = torch.matmul(action_embedding, pre_qs) # batch x n_pivots x n_actions x 1
        qs = qs[:, :, :, 0] # batch x n_pivots x n_actions

        # state_values = torch.max(qs, dim=2, keepdim=False)[0] # batch x n_pivots
        # state_values = torch.mean(qs, dim=2, keepdim=False) # batch x n_pivots; use means stabilize training
        # use mean of top k instead
        # top_k = min(4, n_actions)
        # state_values, _ = torch.topk(qs, top_k, dim=2, largest=True, sorted=False)
        # state_values = torch.mean(state_values, dim=2, keepdim=False)
        # use softmax
        with torch.no_grad():
            sfm = torch.nn.functional.softmax(qs, dim=2)
        state_values = torch.sum(sfm * qs, dim=2, keepdim=False)

        return scores, state_values
