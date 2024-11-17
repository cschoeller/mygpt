import torch
from torch import nn

from base_language_model import BaseLanguageModel


# class SelfAttention(nn.Module):

#     def __init__(self, embed_dim: int):
#         self._qkv_embed = nn.Linear(embed_dim, 3 * embed_dim)

#     def forward(self, tokens: torch.Tensor):
#         # tokens (B, T, C)
#         d = tokens.shape[-1]
#         qkv = self._qkv_embed(tokens).view()
#         q, k, v




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
        pos_embed = self._positional_embed(torch.arange(T, device=token_indices.device)) # (T, C)
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