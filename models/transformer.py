import torch
from torch import nn

from base_language_model import BaseLanguageModel

# enable flash attention if available
# if torch.backends.cuda.is_flash_attention_available():
#     torch.backends.cuda.enable_flash_sdp(enabled=True)


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim: int, heads: int = 1, dim_heads: int | None = None, apply_mask: bool = False):
        super().__init__()
        self._heads = heads
        self._dim_heads = dim_heads if dim_heads is not None else embed_dim
        self._apply_mask = apply_mask
        self._d = nn.parameter.Buffer(torch.sqrt(torch.tensor(embed_dim)))

        # usually can be computed with one linear layer for qkv, but not with this api
        self._heads_q = nn.Linear(embed_dim, heads * self._dim_heads) 
        self._heads_k = nn.Linear(embed_dim, heads * self._dim_heads)
        self._heads_v = nn.Linear(embed_dim, heads * self._dim_heads)
        self._soft = nn.Softmax(dim=-1)
        self._head_to_embed = nn.Linear(heads * self._dim_heads, embed_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # q, k, v of shape (B, T, C)

        # embed into the specified number of heads
        B, T, _ = q.shape
        q = self._heads_q(q).view(B, self._heads, T, self._dim_heads)
        k = self._heads_k(k).view(B, self._heads, T, self._dim_heads)
        v = self._heads_v(v).view(B, self._heads, T, self._dim_heads)

        # apply attention and squash heads
        attn = q @ k.transpose(-1, -2) / self._d
        attn = self._mask_attention(attn)
        attn = self._soft(attn)
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
        self._qkv_embed = nn.Linear(embed_dim, 3 * embed_dim)
        self._attn = MultiHeadAttention(embed_dim=embed_dim, heads=heads, dim_heads=dim_heads, apply_mask=apply_mask)

    def forward(self, tokens: torch.Tensor):
        # tokens (B, T, C)
        qkv = self._qkv_embed(tokens)
        q, k, v = torch.tensor_split(qkv, 3, dim=-1)
        return self._attn(q, k, v)




class Transformer(BaseLanguageModel):
    """Stores a lookup table of embeddings per token to model next-token distributions."""

    def __init__(self, vocab_size: int, context_length: int, embed_dim: int):
        """Initialize the model."""
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self._positional_embed = nn.Embedding(context_length, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_indices: torch.Tensor):
        """Computes the output logits.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        _, T = token_indices.shape
        token_embed = self._token_embedding_table(token_indices) # (B, T, C)
        pos_embed = self._positional_embed(torch.arange(T, device=token_indices.device)) # (T, C) TODO maybe invert order? N,...,0?
        tokens = token_embed + pos_embed
        logits = self.lm_head(tokens) # (B, T, vocab_size)
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
                last_tokens = token_indices[:, -1]
                token_logits = self(last_tokens)
                token_probs = torch.softmax(token_logits/temp, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs, num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices


# transformer = Transformer(12, 10, 32)
# x = torch.randint(0, 12, size=(2, 10))
# y = transformer(x)
# print(y.shape)


# x = torch.randn(size=(2, 10, 5))
# self_attn = SelfAttention(embed_dim=5, heads=3, dim_heads=None, apply_mask=True)
# y = self_attn(x)
# print(x.shape, y.shape)