import torch
from torch import nn

from models.base_language_model import BaseLanguageModel


class BigramLM(BaseLanguageModel):
    """Stores a lookup table of embeddings per token to model next-token distributions."""

    def __init__(self, vocab_size):
        """Initialize the model."""
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_indices: torch.Tensor):
        """Computes the output logits.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        return self._token_embedding_table(token_indices)

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
