import torch
from torch import nn

from models.base_language_model import BaseLanguageModel

# enable flash attention if available
# if torch.backends.cuda.is_flash_attention_available():
#     torch.backends.cuda.enable_flash_sdp(enabled=True)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, heads: int = 1, dim_heads: int | None = None, apply_mask: bool = False):
        super().__init__()
        self._heads = heads
        self._dim_heads = dim_heads if dim_heads is not None else embed_dim
        self._apply_mask = apply_mask
        self._d = nn.parameter.Buffer(torch.tensor(embed_dim**-0.5))

        # usually can be computed with one linear layer for qkv, but not with this api
        self._heads_q = nn.Linear(embed_dim, heads * self._dim_heads, bias=False) 
        self._heads_k = nn.Linear(embed_dim, heads * self._dim_heads, bias=False)
        self._heads_v = nn.Linear(embed_dim, heads * self._dim_heads, bias=False)
        self._softmax = nn.Softmax(dim=-1)
        self._head_to_embed = nn.Linear(heads * self._dim_heads, embed_dim, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q, k, v of shape (B, T, C)

        # embed into the specified number of heads
        B, T, _ = q.shape
        q = self._heads_q(q).view(B, self._heads, T, self._dim_heads)
        k = self._heads_k(k).view(B, self._heads, T, self._dim_heads)
        v = self._heads_v(v).view(B, self._heads, T, self._dim_heads)

        # apply attention and squash heads
        attn = q @ k.transpose(-1, -2) * self._d
        attn = self._mask_attention(attn)
        attn = self._softmax(attn)
        out_heads = attn @ v # (B, heads, T, dim_heads)

        # squash heads
        out_heads = out_heads.transpose(2, 1) # (B, T, heads, dim_heads)
        out_heads = out_heads.flatten(start_dim=2) # flatten heads
        return self._head_to_embed(out_heads)
    
    def _mask_attention(self, attn: torch.Tensor):
        if not self._apply_mask:
            return attn
        
        T = attn.shape[-1]
        tril = torch.tril(torch.ones(size=(T, T), device=attn.device))
        attn = attn.masked_fill(tril == 0., value=float("-inf"))
        return attn


class SelfAttention(nn.Module):

    def __init__(self, embed_dim: int, heads: int = 1, dim_heads: int | None = None, apply_mask: bool = False):
        super().__init__()
        self._qkv_embed = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self._attn = MultiHeadAttention(embed_dim=embed_dim, heads=heads, dim_heads=dim_heads, apply_mask=apply_mask)

    def forward(self, tokens: torch.Tensor):
        # tokens (B, T, C)
        qkv = self._qkv_embed(tokens)
        q, k, v = torch.tensor_split(qkv, 3, dim=-1)
        return self._attn(q, k, v)


class FeedForward(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self._net = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                  nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        return self._net(x)
    

class TransformerLayer(nn.Module):

    def __init__(self, embed_dim: int, heads: int):
        super().__init__()
        self._pre_norm = nn.LayerNorm(embed_dim)
        self._sa = SelfAttention(embed_dim=embed_dim, heads=heads, dim_heads=embed_dim//heads, apply_mask=True)
        self._ffnet = FeedForward(embed_dim=embed_dim)
    
    def forward(self, tokens: torch.Tensor):
        out_tokens = self._pre_norm(tokens)
        out_tokens = self._sa(out_tokens)
        return tokens + self._ffnet(out_tokens)


class Transformer(BaseLanguageModel):
    """Stores a lookup table of embeddings per token to model next-token distributions."""

    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, heads: int, layers: int):
        """Initialize the model."""
        super().__init__()
        self._context_length = context_length
        self._token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self._positional_embed = nn.Embedding(context_length, embed_dim)
        self._sa_head = nn.Sequential(*(TransformerLayer(embed_dim=embed_dim, heads=heads)
                                      for _ in range(layers)))
        self._post_norm = nn.LayerNorm(embed_dim)
        self._lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_indices: torch.Tensor):
        """Computes the output logits.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        *_, T = token_indices.shape
        token_embed = self._token_embedding_table(token_indices) # (B, T, C)
        pos_embed = self._positional_embed(torch.arange(T, device=token_indices.device)) # (T, C) TODO maybe invert order?
        tokens = token_embed + pos_embed
        tokens = self._sa_head(tokens)
        tokens = self._post_norm(tokens)
        logits = self._lm_head(tokens) # (B, T, vocab_size)
        return logits

    def generate(self, token_indices: torch.Tensor, max_new_tokens: int, temp: float = 1.0) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).
            max_new_tokens: Number of next tokens to predict.
            temp: Temperature for the softmax. Lower means focus on highest likelihood, higher more diverse.

        Returns:
            Concatenation of the input tokens with predicted ones, shape (B, T + max_new_tokens).
        """

        with torch.no_grad():

            for _ in range(max_new_tokens):
                last_tokens = token_indices[:, -self._context_length:]
                token_logits = self(last_tokens)
                pred_logits = token_logits[:, -1]
                token_probs = torch.softmax(pred_logits/temp, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs, num_samples=1)
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices

# TODO:
# Feed forward between softmax
# Layer normalization
# Bug in generation? Looks off compared to loss.