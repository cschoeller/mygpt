import torch
import torch.nn as nn

class BigramLanguageModel(nn.Module):
    """Stores a lookup table of embeddings per token"""

    def __init__(self, vocab_size):
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_indices: torch.Tensor):
        """Computes the output logits.
        
        Args:
            token_indices: List of token indices, shape (B, T).

        Returns:
            Logits of shape (B, T, C). Where B is batch, T is time in
            the sequence and C channels (or features).
        """
        return self._token_embedding_table(token_indices)

    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.
        
        Args:
            token_indices: Input token sequences of shape (B, T).
            max_new_tokens: Number of next tokens to predict.

        Returns:
            Concatenation of the input tokens with predicted ones, shape (B, T + max_new_tokens).
        """

        with torch.no_grad():

            for _ in range(max_new_tokens):
                last_tokens = token_indices[:, -1]
                token_logits = self(last_tokens)
                token_probs = torch.softmax(token_logits, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs, num_samples=1)
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices
