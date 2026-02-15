"""
rnn_language_model.py

Implements a character-level LSTM language model.

Architecture:
Embedding → LSTM → Linear → Softmax (via CrossEntropyLoss)
"""

import torch
import torch.nn as nn


class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        Args:
            vocab_size (int): number of unique characters
            embedding_dim (int): size of token embeddings
            hidden_dim (int): hidden state size of LSTM
            num_layers (int): number of stacked LSTM layers
            dropout (float): dropout probability
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Step 1: Embedding layer
        # Converts integer token IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Step 2: LSTM layer
        # batch_first=True → input shape = (batch, seq_len, features)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Step 3: Output projection
        # Maps hidden state → vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, sequence_length)
            hidden: optional initial hidden state

        Returns:
            logits: (batch_size, sequence_length, vocab_size)
            hidden: final hidden state
        """

        # Convert tokens to embeddings
        x = self.embedding(x)
        # Shape: (batch, seq_len, embedding_dim)

        # Pass through LSTM
        output, hidden = self.lstm(x, hidden)
        # output shape: (batch, seq_len, hidden_dim)

        # Project each timestep to vocabulary space
        logits = self.fc(output)
        # logits shape: (batch, seq_len, vocab_size)

        return logits, hidden

    def generate(self, x, max_new_tokens):
        self.eval()

        hidden = None

        for _ in range(max_new_tokens):

            logits, hidden = self(x, hidden)

            # take last time step
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, next_token), dim=1)

        return x

