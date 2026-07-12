import torch
import torch.nn as nn


class InvertedDataEmbedding(nn.Module):
    """Embed each PV site sequence as one token and append time-feature tokens."""

    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        tokens = x.permute(0, 2, 1).contiguous()
        if x_mark is not None:
            time_tokens = x_mark.permute(0, 2, 1).contiguous()
            tokens = torch.cat([tokens, time_tokens], dim=1)
        return self.dropout(self.value_embedding(tokens))
