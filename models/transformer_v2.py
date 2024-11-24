import torch
from torch import nn

from models.base_language_model import BaseLanguageModel
torch.set_printoptions(profile="full", sci_mode=False, precision=2)

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim: int, heads: int, drop_rate: float):
        super().__init__()
        attn_mask = torch.triu(torch.ones(256, 256, dtype=torch.float32) * float('-inf'), diagonal=1)
        self._attn_mask = nn.Buffer(attn_mask, persistent=True)
        self._attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=drop_rate,
                                           add_bias_kv=False, batch_first=True)
        self._dropout = nn.Dropout(drop_rate)

    def forward(self, tokens: torch.Tensor):
        # tokens (B, T, C)
        q, k, v = tokens, tokens, tokens
        # uses flash attention is available and enabled
        out, _ = self._attn(q, k, v, attn_mask=self._attn_mask, is_causal=True, need_weights=False)
        return self._dropout(out)


class FeedForward(nn.Module):

    def __init__(self, embed_dim: int, drop_rate: float):
        super().__init__()
        self._net = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim),
                                  nn.ReLU(),
                                  nn.Linear(4 * embed_dim, embed_dim),
                                  nn.Dropout(drop_rate))
    
    def forward(self, x: torch.Tensor):
        return self._net(x)
    

class TransformerLayer(nn.Module):

    def __init__(self, embed_dim: int, heads: int, drop_rate: float):
        super().__init__()
        self._ln1 = nn.LayerNorm(embed_dim)
        self._ln2 = nn.LayerNorm(embed_dim)
        self._sa = CausalSelfAttention(embed_dim=embed_dim, heads=heads, drop_rate=drop_rate)
        self._ffnet = FeedForward(embed_dim=embed_dim, drop_rate=drop_rate)
    
    def forward(self, tokens: torch.Tensor):
        tokens = tokens + self._sa(self._ln1(tokens))
        tokens = tokens + self._ffnet(self._ln2(tokens))
        return tokens


class TransformerV2(BaseLanguageModel):
    """Stores a lookup table of embeddings per token to model next-token distributions."""

    def __init__(self, vocab_size: int, context_length: int, embed_dim: int, heads: int, n_layers: int, drop_rate: float = 0.0):
        """Initialize the model."""
        super().__init__()
        self._context_length = context_length
        self._token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self._positional_embed = nn.Embedding(context_length, embed_dim)
        self._sa_head = nn.Sequential(*(TransformerLayer(embed_dim=embed_dim, heads=heads, drop_rate=drop_rate)
                                      for _ in range(n_layers)))
        self._norm = nn.LayerNorm(embed_dim)
        self._lm_head = nn.Linear(embed_dim, vocab_size)
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
        token_embed = self._token_embedding_table(token_indices) # (B, T, C)
        token_pos_indices = torch.arange(T, device=token_indices.device)
        pos_embed = self._positional_embed(token_pos_indices) # (T, C)
        tokens = token_embed + pos_embed # (B, T, embed_dim)
        tokens = self._sa_head(tokens) # (B, T, embed_dim)
        tokens = self._norm(tokens) # (B, T, embed_dim)
        return self._lm_head(tokens) # (B, T, vocab_size)

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
            last_tokens = token_indices[:, -self._context_length:]
            token_logits = self(last_tokens)
            pred_logits = token_logits[:, -1]
            token_probs = torch.softmax(pred_logits/temp, dim=-1)
            pred_tokens = torch.multinomial(input=token_probs, num_samples=1)
            token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

        return token_indices
