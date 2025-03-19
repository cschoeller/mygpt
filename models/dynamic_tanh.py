import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    """
    Dynamic tanh normalization function.

    Based on the paper "Transformers without Normalization". The normalization operates
    element-wise and the idea is that usually layernorm follows a tanh-like shape and can
    be replaced directly by such a function.

    Link: https://arxiv.org/abs/2503.10622
    """

    def __init__(self, embed_dim: int, alpah_init: float = 0.5):
        super().__init__()
        self._alpha = nn.Parameter(torch.ones(1) * alpah_init)
        self._gamma = nn.Parameter(torch.ones(embed_dim))
        self._beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        """Applies the element-wise normalization."""
        x = torch.tanh(self._alpha * x)
        return self._gamma * x + self._beta
