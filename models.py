from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseLanguageModel(nn.Module, ABC):

    @abstractmethod
    def forward(self, token_indices: torch.Tensor):
        ...
    
    @abstractmethod
    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        ...


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

    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).
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
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices


class RecurrentLM(BaseLanguageModel):
    """GRU based recurrent language prediction model."""

    def __init__(self, vocab_size: int, *, embed_dim: int = 16, hidden_dim: int = 64, num_layers: int = 3):
        """Initialize the model.
        
        Args:
            vocab_size: Size of the token vocabulary.
            embed_dim: Dims of token embeddings.
            hidden_size: Size of the GRUs internal hidden states.
            num_layers: Number of layers in the stacked GRU.
        """
        super().__init__()
        self._token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self._rnn = torch.nn.GRU(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self._h0 = torch.nn.Parameter(torch.rand(size=(num_layers, 1, hidden_dim), dtype=torch.float32))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, vocab_size))

    def forward(self, token_indices: torch.Tensor):
        """Compute the logits for each token in the sequences.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        encoded_tokens = self._token_embeddings(token_indices)
        h0 = self._h0.expand(-1, token_indices.shape[0] , -1)
        outputs, _ = self._rnn(encoded_tokens, h0)
        return self.decoder(outputs)

    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).
            max_new_tokens: Number of next tokens to predict.

        Returns:
            Concatenation of the input tokens with predicted ones, shape (B, T + max_new_tokens).
        """
        with torch.no_grad():

            for _ in range(max_new_tokens):
                last_tokens = token_indices[:, -1]
                token_logits = self(last_tokens.unsqueeze(-1))
                token_probs = torch.softmax(token_logits, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs.squeeze(), num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices


class RecurrentLMGraves(BaseLanguageModel):
    """GRU based recurrent language, but with inputs fed to every cell.
    
    Inspired by Graves et al., 2014. URL: https://arxiv.org/abs/1308.0850.
    """

    def __init__(self, vocab_size: int, *, embed_dim: int = 16, hidden_dim: int = 64, num_layers: int = 3):
        """Initialize the model.
        
        Args:
            vocab_size: Size of the token vocabulary.
            embed_dim: Dims of token embeddings.
            hidden_dim: Size of the GRUs internal hidden states.
            num_layers: Number of layers in the stacked GRU.
        """
        super().__init__()
        self._num_layers = num_layers
        self._token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self._rnn_cells = nn.ModuleList(nn.GRUCell(embed_dim + hidden_dim, hidden_size=hidden_dim) for _ in range(num_layers))
        self._h0 = torch.nn.Parameter(torch.rand(size=(num_layers, hidden_dim), dtype=torch.float32))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, max(hidden_dim//2, vocab_size)),
                                     nn.ReLU(),
                                     nn.Linear(max(hidden_dim//2, vocab_size), vocab_size),
                                     )


    def forward(self, token_indices: torch.Tensor):
        """Compute the logits for each token in the sequences.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        encoded_tokens = self._token_embeddings(token_indices).permute(1, 0, 2) # seq first

        outputs = []
        curr_cell_h = [h.unsqueeze(0).expand(token_indices.shape[0], -1) for h in self._h0]

        # iterate sequence
        for seq_idx in range(len(encoded_tokens)):
            curr_token_embed = encoded_tokens[seq_idx]

            # iterate gru cells (layers)
            for layer_idx in range(self._num_layers):

                # build input as cat of token embedding and hidden cell state
                h_cell = curr_cell_h[layer_idx]
                hx = torch.cat([h_cell, curr_token_embed], dim=-1)

                # compute new hidden cell state and update in memory
                h_new = self._rnn_cells[layer_idx](hx)
                curr_cell_h[layer_idx] = h_new

                # decode and store hidden state of last layer
                if layer_idx == self._num_layers - 1:
                    outputs.append(self.decoder(h_new))

        return torch.stack(outputs, dim=1)

    def generate(self, token_indices: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Samples max_new_tokens predicted tokens based on the input tokens.
        
        Args:
            token_indices: Sequences of tokens, shape (B, T).
            max_new_tokens: Number of next tokens to predict.

        Returns:
            Concatenation of the input tokens with predicted ones, shape (B, T + max_new_tokens).
        """
        with torch.no_grad():

            for _ in range(max_new_tokens):
                last_tokens = token_indices[:, -1]
                token_logits = self(last_tokens.unsqueeze(-1))
                token_probs = torch.softmax(token_logits, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs.squeeze(), num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

            return token_indices
