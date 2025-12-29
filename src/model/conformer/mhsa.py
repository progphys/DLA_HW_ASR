import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(T, C, device, dtype):
    position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, C, 2, device=device, dtype=dtype) * (-math.log(10000.0) / C)
    )
    pe = torch.zeros(T, C, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class MHSAModule(nn.Module):
    def __init__(self, in_features, n_heads, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.mha = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        x = self.ln(x)

        B, T, C = x.shape
        pe = sinusoidal_positional_encoding(T, C, device=x.device, dtype=x.dtype)
        x = x + pe
        y, _ = self.mha(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.drop(y)
