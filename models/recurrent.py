import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_language_model import BaseLanguageModel


class RecurrentLM(BaseLanguageModel):
    """GRU based recurrent language prediction model."""

    def __init__(self, vocab_size: int, *, hidden_dim: int = 64, num_layers: int = 3):
        """Initialize the model.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_size: Size of the GRUs internal hidden states.
            num_layers: Number of layers in the stacked GRU.
        """
        super().__init__()
        self._vocab_size = vocab_size
        self._rnn = torch.nn.GRU(input_size=vocab_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, vocab_size))

    def forward(
        self, token_indices: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the logits for each token in the sequences.

        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            outputs: Logits of shape (B, T, V). Where B is batch, T is time in
                the sequence and V is the vocabulary size.
            hidden: Hidden state of hsape (B, L, H), i.e., the per batch the latest
                hidden state of each gru layer..
        """
        encoded_tokens = F.one_hot(token_indices, num_classes=self._vocab_size).float()
        outputs, hidden = self._rnn(encoded_tokens, hidden)

        if self.training:
            return self.decoder(outputs)

        return self.decoder(outputs), hidden

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
        assert not self.training, "No text generation in training mode."
        input_seq_length = token_indices.shape[1]

        hidden = None
        for i in range(input_seq_length + max_new_tokens):
            next_token = token_indices[:, i]
            token_logits, hidden = self(next_token.unsqueeze(-1), hidden)

            if i >= input_seq_length - 1:  # start generating
                token_probs = torch.softmax(token_logits / temp, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs.squeeze(), num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

        return token_indices


class RecurrentLMGraves(BaseLanguageModel):
    """GRU based recurrent language, but with inputs fed to every cell.

    Inspired by Graves et al., 2014. URL: https://arxiv.org/abs/1308.0850.
    """

    def __init__(self, vocab_size: int, *, hidden_dim: int = 64, num_layers: int = 3):
        """Initialize the model.

        Args:
            vocab_size: Size of the token vocabulary.
            hidden_dim: Size of the GRUs internal hidden states.
            num_layers: Number of layers in the stacked GRU.
        """
        super().__init__()
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self._rnn_cells = nn.ModuleList(nn.GRUCell(vocab_size, hidden_size=hidden_dim) for _ in range(num_layers))
        self._linear_prev = nn.Linear(hidden_dim, hidden_dim)
        self._linear_curr = nn.Linear(hidden_dim, hidden_dim)
        self._h0 = torch.nn.Parameter(torch.rand(size=(num_layers, hidden_dim), dtype=torch.float32))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, vocab_size)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, vocab_size), vocab_size),
        )

    def forward(
        self, token_indices: torch.Tensor, hidden: list[torch.Tensor] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the logits for each token in the sequences.

        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        encoded_tokens = F.one_hot(token_indices, num_classes=self._vocab_size).float().permute(1, 0, 2)  # seq first
        # encoded_tokens = self._token_embeddings(token_indices).permute(1, 0, 2).contiguous() # seq first

        outputs = []
        curr_cell_h = (
            hidden if hidden is not None else [h.unsqueeze(0).expand(token_indices.shape[0], -1) for h in self._h0]
        )

        # iterate sequence
        for seq_idx in range(len(encoded_tokens)):
            curr_token_embed = encoded_tokens[seq_idx]

            # iterate gru cells (layers)
            for layer_idx in range(self._num_layers):
                # build hidden state input by adding state of previous layer
                h_prev = self._linear_prev(curr_cell_h[layer_idx - 1]) if layer_idx >= 1 else 0.0
                h = self._linear_curr(curr_cell_h[layer_idx])
                hx = h + h_prev

                # compute new hidden cell state and update in memory
                h_new = self._rnn_cells[layer_idx](curr_token_embed, hx)
                curr_cell_h[layer_idx] = h_new

                # decode and store hidden state of last layer
                if layer_idx == self._num_layers - 1:
                    # average pool the hidden states of all layers, skip connections
                    hidden_stack = torch.stack(curr_cell_h, dim=-1)  # (B, hidden_dim, L)
                    # aggregated_hidden = torch.mean(hidden_stack, dim=-1) # (B, hidden_dim)
                    aggregated_hidden, _ = torch.max(hidden_stack, dim=-1)  # (B, hidden_dim)
                    outputs.append(self.decoder(aggregated_hidden))

        if self.training:
            return torch.stack(outputs, dim=1)

        return torch.stack(outputs, dim=1), curr_cell_h

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
        assert not self.training, "No text generation in training mode."
        input_seq_length = token_indices.shape[1]

        hidden = None
        for i in range(input_seq_length + max_new_tokens):
            next_token = token_indices[:, i]
            token_logits, hidden = self(next_token.unsqueeze(-1), hidden)

            if i >= input_seq_length - 1:  # start generating
                token_probs = torch.softmax(token_logits / temp, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs.squeeze(), num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

        return token_indices


class RecurrentEnsembleLM(BaseLanguageModel):
    """GRU based recurrent language, but with inputs fed to every cell."""

    def __init__(
        self,
        vocab_size: int,
        *,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_fcn_hidden: bool = False,
    ):
        """Initialize the model.

        Args:
            vocab_size: Size of the token vocabulary.
            embed_dim: Dims of token embeddings.
            hidden_dim: Size of the GRUs internal hidden states.
            num_layers: Number of layers in the stacked GRU.
            use_fcn_hidden: Process hidden states with a small fcn.
        """
        super().__init__()
        self._num_layers = num_layers
        self._use_fcn_hidden = use_fcn_hidden
        self._token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self._rnn_cells = nn.ModuleList(nn.GRUCell(embed_dim, hidden_size=hidden_dim) for _ in range(num_layers))
        self._fcn_connectors = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers)
        )
        self._h0 = torch.nn.Parameter(torch.rand(size=(num_layers, hidden_dim), dtype=torch.float32))
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, vocab_size)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, vocab_size), vocab_size),
        )

    def forward(
        self, token_indices: torch.Tensor, hidden: list[torch.Tensor] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute the logits for each token in the sequences.

        Args:
            token_indices: Sequences of tokens, shape (B, T).

        Returns:
            Logits of shape (B, T, V). Where B is batch, T is time in
            the sequence and V is the vocabulary size.
        """
        encoded_tokens = self._token_embeddings(token_indices).permute(1, 0, 2)  # seq first

        outputs = []
        curr_cell_h = (
            hidden if hidden is not None else [h.unsqueeze(0).expand(token_indices.shape[0], -1) for h in self._h0]
        )

        # iterate sequence
        for seq_idx in range(len(encoded_tokens)):
            curr_token_embed = encoded_tokens[seq_idx]

            # iterate gru cells (layers)
            for layer_idx in range(self._num_layers):
                # build input as cat of token embedding and hidden cell state
                hx = curr_cell_h[layer_idx]
                if self._use_fcn_hidden:
                    hx = self._fcn_connectors[layer_idx](hx)  # dense transform

                # compute new hidden cell state and update in memory
                h_new = self._rnn_cells[layer_idx](curr_token_embed, hx)
                curr_cell_h[layer_idx] = h_new

                # decode and store hidden state of last layer
                if layer_idx == self._num_layers - 1:
                    # average pool the hidden states of all layers, skip connections
                    hidden_stack = torch.stack(curr_cell_h, dim=-1)  # (B, hidden_dim, L)
                    aggregated_hidden, _ = torch.max(hidden_stack, dim=-1)  # (B, hidden_dim)
                    outputs.append(self.decoder(aggregated_hidden))

                    # old version, only used last cell's hidden state to predict
                    # outputs.append(self.decoder(h_new))

        if self.training:
            return torch.stack(outputs, dim=1)

        return torch.stack(outputs, dim=1), curr_cell_h

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
        assert not self.training, "No text generation in training mode."
        input_seq_length = token_indices.shape[1]

        hidden = None
        for i in range(input_seq_length + max_new_tokens):
            next_token = token_indices[:, i]
            token_logits, hidden = self(next_token.unsqueeze(-1), hidden)

            if i >= input_seq_length - 1:  # start generating
                token_probs = torch.softmax(token_logits / temp, dim=-1)
                pred_tokens = torch.multinomial(input=token_probs.squeeze(), num_samples=1)
                pred_tokens = pred_tokens.unsqueeze(0) if len(pred_tokens.shape) == 1 else pred_tokens
                token_indices = torch.cat([token_indices, pred_tokens], dim=-1)

        return token_indices
