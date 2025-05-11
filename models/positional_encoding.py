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

    def forward(self, tokens: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
        return tokens + self._pos_enc[pos_indices]


class LearnedPositionalEncoding(nn.Module):
    """Positional encoding with learned positional embeddings."""

    def __init__(self, context_length: int, embed_dim: int):
        super().__init__()
        self._pos_enc = nn.Embedding(context_length, embed_dim)

    def forward(self, tokens: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
        return tokens + self._pos_enc(pos_indices)


class NoPos(nn.Module):
    """No positional encoding.

    Based on the observations in paper 'Transformer Language Models without Positional
    Encodings Still Learn Positional Information', Haviv et al., 2022.
    LinK: https://arxiv.org/abs/2203.16634

    Using it the model still reaches a good loss, but the output text seems less coherent.
    This could be because I didn't train long enough in my tests or the model is too small
    thus and not able to learn positional information.
    """

    def forward(self, tokens: torch.Tensor, pos_indices: torch.Tensor) -> torch.Tensor:
        """Return the input tokens unchanged, ignoring positional indices."""
        return tokens


class RotaryPosEmbedding(nn.Module):
    """Rotary positional embedding.

    "RoFormer: Enhanced Transformer with Rotary Position Embedding", Su et al., 2024.

    NOTE:
        1. There is a more efficient way to compute this, by chunking the token dimensions into
        half and rotating [x_i, x_(N//2 +i)] instead of [x_i, x_(i+1)]. It can be considered mathematically
        equivalent, just differently permuted.
        See: https://github.com/KellerJordan/modded-nanogpt/blob/64d8eb51ee951e09d9dba09269e0ddc6a099b9a9/train_gpt.py#L233
        Pseudo code:
            x1, x2 = tokens.chunk(2, dim=-1)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), dim=-1)
        2. For very long context lengths the pre-computing and caching becomes expensive. In this case we could compute the
        outer product for those positions we need on the fly and only cache the angles.
    """

    def __init__(self, embed_dim: int, max_seq_len: int, theta_base: float = 1e4, p: float = 1.0):
        """Initialize the positional encoding.

        Args:
            embed_dim: The embedding dimension.
            max_seq_len: The maximum sequence length.
            theta_base: The base for computing the angles.
            p: The p-RoPe percentage, i.e., the percentage of features to use for RoPe.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")

        if p < 0 or p > 1:
            raise ValueError("p must be between 0 and 1")

        self._embed_dim = embed_dim
        self._max_seq_len = max_seq_len
        self._max_idx = int(embed_dim // 2 * p) * 2

        # pre-compute the rotations
        self._pos_indices = nn.Buffer(torch.arange(max_seq_len, dtype=torch.long), persistent=False)
        theta = theta_base ** (-2.0 * (torch.arange(embed_dim // 2, dtype=torch.float32)) / embed_dim)
        angle_matrix = torch.outer(self._pos_indices, theta)  # multiply every position with every angle
        self._cos_cache = nn.Buffer(angle_matrix.cos().repeat_interleave(2, dim=-1), persistent=False)
        self._sin_cache = nn.Buffer(angle_matrix.sin().repeat_interleave(2, dim=-1), persistent=False)
        self._sin_cache[:, 0::2] *= -1  # negate the even indices

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Rotate the input tokens by their positional index.

        The input tokens should have shape (B, T, H, embed_dim), where B is the batch size,
        T is the sequence length, H is the number of heads, and embed_dim is the embedding
        dimension.

        Args:
            tokens: Input tokens to encode.

        """
        if len(tokens.shape) != 4:
            raise ValueError("The input tokens must have shape (B, T, H, embed_dim)")

        if tokens.shape[-1] != self._embed_dim:
            raise ValueError("The last dimension of tokens must be equal to embed_dim")

        _, seq_len, *_ = tokens.shape
        if seq_len > self._max_seq_len:
            raise ValueError("The maximum sequence length is exceeded.")

        tokens_rot = tokens[..., : self._max_idx]

        pos_indices = self._pos_indices[:seq_len]
        tokens_rot_perm = tokens_rot.view(*tokens_rot.shape[:-1], tokens_rot.shape[-1] // 2, 2)[..., [1, 0]].flatten(-2)

        tokens_rot = (
            tokens_rot * self._cos_cache[pos_indices, None, : self._max_idx].to(tokens.dtype)  # buffer mixed precision
            + tokens_rot_perm * self._sin_cache[pos_indices, None, : self._max_idx].to(tokens.dtype)
        )

        return torch.cat(
            [
                tokens_rot,
                tokens[..., self._max_idx :],
            ],
            dim=-1,
        )
