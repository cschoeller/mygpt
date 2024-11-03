import torch.nn as nn

class BigramLanguageModel(nn.Module):
    """Stores a lookup table of embeddings per token"""

    def __init__(self, vocab_size):
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_indices):
        """Computes the output logits.
        
        Args:
            token_indices: List of token sindices, shape (B, T).
            targets: 

        Returns:
            Logits of shape (B, T, C). Where B is batch, T is time in
            the sequence and C channels (or features).
        """
        return self._token_embedding_table(token_indices)
