import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding of 'Attention is all you need', Vaswani et al., 2017."""

    def __init__(self, context_length: int, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"

        pos = torch.arange(context_length, dtype=torch.float32)
        even_dims = torch.arange(0, embed_dim, 2, dtype=torch.float32)
        scale_factors = 10000.0 ** (-even_dims / embed_dim)
        scaled_pos = pos.unsqueeze(1) * scale_factors

        # compute the positional encodings
        pos_enc = torch.zeros((context_length, embed_dim))
        pos_enc[:, 0::2] = torch.sin(scaled_pos)  # even dims
        pos_enc[:, 1::2] = torch.cos(scaled_pos)  # odd dims

        self._pos_enc = nn.Buffer(pos_enc, persistent=False)

    def forward(self, pos_indices: torch.Tensor) -> torch.Tensor:
        return self._pos_enc[pos_indices]
