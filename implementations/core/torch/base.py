import torch
import torch.nn as nn
import torch.nn.functional as F


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


def softmax_with_temperature(logits, temperature=1.0, dim=-1):
    """
    Numerically stable softmax with temperature scaling.

    Args:
        logits (torch.Tensor): The input tensor of logits.
        temperature (float, optional): The temperature parameter. Defaults to 1.0.
        dim (int, optional): The dimension along which to compute softmax. Defaults to -1.

    Returns:
        torch.Tensor: The softmax output with temperature scaling.
    """
    logits = logits / temperature
    max_values = torch.max(logits, dim=dim, keepdim=True)[0]
    shifted_logits = logits - max_values
    
    exp_values = torch.exp(shifted_logits)
    sum_exp_values = torch.sum(exp_values, dim=dim, keepdim=True)
    
    softmax_output = exp_values / sum_exp_values
    
    return softmax_output


class Entropy_Function(torch.autograd.Function):
    """
    Custom Entropy function with manually defined gradient.
    Entropy: H(p) = - sum(p * log(p)).
    Avoids p * log(p) in the backward pass for stability or specific numerical reasons.
    """
    @staticmethod
    def forward(ctx, p, dim):
        epsilon = 1e-12
        p_stable = p + epsilon
        log_p = torch.log(p_stable)
        entropy = -torch.sum(p * log_p, dim=dim)

        # Save p_stable for backward pass (or log_p directly)
        ctx.save_for_backward(p_stable)
        return entropy

    @staticmethod
    def backward(ctx, grad_output):
        p_stable, = ctx.saved_tensors # Retrieve saved tensor
        dim = ctx.dim
        grad_entropy_p = -(torch.log(p_stable) + 1.0)
        grad_input = grad_output.unsqueeze(dim) * grad_entropy_p

        # The 'dim' argument does not require a gradient
        return grad_input, None


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


class Res_Net(Multilayer_Relu):
    def __init__(self, input_size, output_size, hidden_size, n_layers=1, device=None):
        super(Res_Net, self).__init__(input_size, output_size, hidden_size, n_layers, device)

    def forward(self, x):
        for layer in self.layers[:-1]:
            pre_x = x
            x = F.relu(layer(x))
            x = x + pre_x
        x = self.layers[-1](x)
        return x
