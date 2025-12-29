import torch
import torch.nn as nn
from block import ConformerBlock


class ConformerBlocks(nn.Module):
    def __init__(
        self,
        num_layers,
        in_features,
        n_heads,
        fc_features,
        kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    in_features=in_features,
                    n_heads=n_heads,
                    fc_features=fc_features,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, key_padding_mask):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
