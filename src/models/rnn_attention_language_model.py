import torch
import torch.nn as nn


class RNNAttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Additive (Bahdanau-style) attention
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

        # Final projection
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):

        x = self.embedding(x)
        outputs, _ = self.lstm(x)
        # outputs: (B, T, H)

        B, T, H = outputs.shape

        # -------- Vectorized Additive Attention --------

        # Expand for pairwise interaction
        outputs_i = outputs.unsqueeze(2)  # (B, T, 1, H)
        outputs_j = outputs.unsqueeze(1)  # (B, 1, T, H)

        score = torch.tanh(
            self.W1(outputs_i) + self.W2(outputs_j)
        )

        score = self.V(score).squeeze(-1)  # (B, T, T)

        # -------- Causal Mask (VERY IMPORTANT) --------
        mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = mask.unsqueeze(0)  # (1, T, T)

        score = score.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(score, dim=-1)

        # Compute context
        context = torch.matmul(attn_weights, outputs)  # (B, T, H)

        # Concatenate context with original outputs
        combined = torch.cat([outputs, context], dim=-1)

        logits = self.fc(combined)

        return logits
