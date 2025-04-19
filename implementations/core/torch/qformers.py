import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Multilayer_Relu, apply_transformer, causal_mask, positional_encoding


def access(Vs, Ss, x, scores, num_slots, dims, value_access):

    Vs = torch.reshape(Vs, (-1, num_slots, dims))
    Ss = torch.reshape(Ss, (-1, num_slots))

    # Access
    if value_access:
        x = torch.reshape(x, (-1, dims))
        Vl = torch.unsqueeze(x, 2)
        denom = torch.norm(Vs, dim=2, keepdims=True) * torch.norm(Vl, dim=1, keepdims=True)
        # prevent divide by zero
        denom = torch.maximum(1e-6, denom)
        dot_scores = torch.matmul(Vs, Vl) / denom
        # force dot_scores shape [batch, memory_size]
        dot_scores = torch.reshape(dot_scores, (-1, num_slots))
        max_indices = torch.argmax(dot_scores, dim=1, keepdims=True)    
    else:
        # score has range [0, 1], quantize to int slots, must be handled from out side
        scores = torch.reshape(scores, (-1, 1))
        max_indices = torch.round(scores * (num_slots - 1)).to(torch.int64)
        max_indices = torch.clamp(max_indices, min=0, max=num_slots - 1)

    v = torch.gather(Vs, torch.unsqueeze(max_indices, dim=-1).expand(-1, -1, dims), axis=1)
    s = torch.gather(Ss, 1, max_indices)
    
    return v, s


class Q_Table(nn.Module):
    def __init__(self, input_size, hidden_size, num_output_qs, device):
        super(Q_Table, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_output_qs = num_output_qs

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

        self.pe = positional_encoding(256, hidden_size).to(device) # 256 is the maximum length of the context


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

        qs = apply_transformer(self.q_decoder, collapsed_actions, memory=collapsed_pivot_state_internal)
        qs = torch.reshape(qs, (batch, n_pivots, n_actions, self.hidden_size)) # batch x n_pivots x n_actions x hidden
        qs = qs[:, :, :, 0] # batch x n_pivots x n_actions

        # state_values = torch.max(qs, dim=2, keepdim=False)[0] # batch x n_pivots
        state_values = torch.mean(qs, dim=2, keepdim=False) # batch x n_pivots; use means stabilize training
        return qs, state_values
