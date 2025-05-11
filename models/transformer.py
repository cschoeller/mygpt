from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from models.base_language_model import BaseLanguageModel
from models.dynamic_tanh import DynamicTanh
from models.positional_encoding import (
    LearnedPositionalEncoding,
    NoPos,
    RotaryPosEmbedding,
    SinusoidalPositionalEncoding,
)
from models.relu2 import ReLU2


@dataclass
class TransformerParams:
    vocab_size: int
    context_length: int = 256
    embed_dim: int = 384
    heads: int = 6
    n_layers: int = 6
    drop_rate: float = 0.2
    u_net_skips: bool = False
    ffn_activation: str = "relu"
    normalization: str = "layernorm"
    positional_encoding: str = "learned"


def _get_activation(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "relu2":
        return ReLU2()
    else:
        raise ValueError(f"Unknown activation: {activation}")


def _get_normalization(normalization: str, embed_dim: int):
    if normalization == "layernorm":
        return nn.LayerNorm(embed_dim)
    elif normalization == "dynamic_tanh":
        return DynamicTanh(embed_dim)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def _get_positional_encoding(positional_encoding: str, context_length: int, embed_dim: int):
    if positional_encoding == "sinusoidal":
        return SinusoidalPositionalEncoding(context_length, embed_dim)
    elif positional_encoding == "learned":
        return LearnedPositionalEncoding(context_length, embed_dim)
    elif positional_encoding == "rotary":  # applied in each attention layer
        return None
    elif positional_encoding == "nopos":
        return NoPos()
    else:
        raise ValueError(f"Unknown positional encoding: {positional_encoding}")


def _get_attention_layer(params: TransformerParams):
    if params.positional_encoding == "rotary":
        return RotaryCausalSelfAttention(
            embed_dim=params.embed_dim,
            num_heads=params.heads,
            drop_rate=params.drop_rate,
            max_seq_len=params.context_length,
        )

    return CausalSelfAttention(
        embed_dim=params.embed_dim,
        num_heads=params.heads,
        drop_rate=params.drop_rate,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, drop_rate: float):
        super().__init__()
        attn_mask = torch.triu(torch.ones(256, 256, dtype=torch.float32) * float("-inf"), diagonal=1)
        self._attn_mask = nn.Buffer(attn_mask, persistent=True)
        self._attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=drop_rate, add_bias_kv=False, batch_first=True
        )
        self._dropout = nn.Dropout(drop_rate)

    def forward(self, tokens: torch.Tensor):
        """Computes the output of the causal self-attention.

        Args:
            tokens: Input tokens, shape (B, T, C).
        """
        # uses flash attention is available and enabled
        q, k, v = tokens, tokens, tokens
        out, _ = self._attn(q, k, v, attn_mask=self._attn_mask, is_causal=True, need_weights=False)
        return self._dropout(out)


class RotaryCausalSelfAttention(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, drop_rate: float, max_seq_len: int = 256):
        super().__init__()
        # self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._max_seq_len = max_seq_len
        self._qkv_embed = nn.Linear(embed_dim, 3 * num_heads * self._head_dim, bias=False)
        self._rotary_embedding = RotaryPosEmbedding(self._head_dim, max_seq_len=max_seq_len)
        self._drop_rate = drop_rate
        self._head_to_embed = nn.Linear(num_heads * self._head_dim, embed_dim, bias=False)
        self._dropout = nn.Dropout(drop_rate)

    def forward(self, tokens: torch.Tensor):
        """Computes the output of the rotary causal self-attention.

        Args:
            tokens: Input tokens, shape (B, T, C).
        """
        # to (B, T, 3 * num_heads, head_dim)
        out_heads = self._qkv_embed(tokens).view(*tokens.shape[:-1], 3 * self._num_heads, self._head_dim)
        q, k, v = out_heads.chunk(3, dim=-2)  # to (B, T, num_heads, head_dim)
        q, k = self._rotary_embedding(q), self._rotary_embedding(k)
        out_heads = scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=self._drop_rate, is_causal=True
        ).transpose(1, 2)  # transpose back to (B, T, num_heads, head_dim)
        out_heads = out_heads.flatten(start_dim=2)  # flatten heads
        out = self._head_to_embed(out_heads)
        return self._dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, drop_rate: float, activation: str = "relu"):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            _get_activation(activation),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor):
        return self._net(x)


class TransformerLayer(nn.Module):
    def __init__(self, params: TransformerParams):
        super().__init__()
        self._norm1 = _get_normalization(params.normalization, params.embed_dim)
        self._norm2 = _get_normalization(params.normalization, params.embed_dim)
        self._sa = _get_attention_layer(params)
        self._ffnet = FeedForward(
            embed_dim=params.embed_dim, drop_rate=params.drop_rate, activation=params.ffn_activation
        )

    def forward(self, tokens: torch.Tensor):
        tokens = tokens + self._sa(self._norm1(tokens))
        tokens = tokens + self._ffnet(self._norm2(tokens))
        return tokens


class Transformer(BaseLanguageModel):
    """Stores a lookup table of embeddings per token to model next-token distributions."""

    def __init__(self, params: TransformerParams):
        """Initialize the model."""
        super().__init__()
        self._context_length = params.context_length
        self._skip_layers = params.u_net_skips
        self._num_layers = params.n_layers
        self._token_embedding_table = nn.Embedding(params.vocab_size, params.embed_dim)
        self._positional_encoder = _get_positional_encoding(
            params.positional_encoding, params.context_length, params.embed_dim
        )
        assert self._positional_encoder == None if params.positional_encoding == "rotary" else True
        self._layers = nn.ModuleList([TransformerLayer(params) for _ in range(self._num_layers)])
        self._norm = _get_normalization(params.normalization, params.embed_dim)
        self._lm_head = nn.Linear(params.embed_dim, params.vocab_size, bias=False)
        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_indices: torch.Tensor):
        """Computes the output logits.

        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        *_, T = token_indices.shape
        tokens = self._token_embedding_table(token_indices)  # (B, T, embed_dim)
        if self._positional_encoder is not None:
            token_pos_indices = torch.arange(T, device=token_indices.device)
            tokens = self._positional_encoder(tokens, token_pos_indices)  # (B, T, embed_dim)

        # compute pass through layers, optionally with skip connections
        intermediate = []
        first_skip_layer_idx = self._num_layers // 2 + 1
        for i, layer in enumerate(self._layers):
            tokens = layer(tokens)

            if self._skip_layers:
                if i >= first_skip_layer_idx:
                    tokens = tokens + intermediate[self._num_layers - 1 - i]

                intermediate.append(tokens)

        return self._lm_head(self._norm(tokens))  # (B, T, vocab_size)

    @torch.no_grad()
    def generate(self, token_indices: torch.Tensor, max_new_tokens: int, temp: float = 1.0) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.

        Args:
            token_indices: Sequences of tokens, shape (B, T).
            max_new_tokens: Number of next tokens to predict.
            temp: Temperature for the softmax. Lower means focus on highest likelihood, higher more diverse.

        Returns:
            Concatenation of the input tokens with predicted ones, shape (B, T + max_new_tokens).
        """

        for _ in range(max_new_tokens):
            last_tokens = token_indices[:, -self._context_length :]
            token_logits = self(last_tokens)
            pred_logits = token_logits[:, -1]
            token_probs = torch.softmax(pred_logits / temp, dim=-1)
            pred_tokens = torch.multinomial(input=token_probs, num_samples=1)
            token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

        return token_indices
