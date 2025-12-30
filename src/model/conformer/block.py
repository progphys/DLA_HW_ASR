import torch
import torch.nn as nn

from .conv_module import ConformerConvModule
from .ffn import FeedForwardModule
from .mhsa import MHSAModule


class ConformerBlock(nn.Module):
    """
    Conformer block:
      x = x + 1/2 FFN(x)
      x = x + MHSA(x)
      x = x + Conv(x)
      x = x + 1/2 FFN(x)
      x = LayerNorm(x)
    """

    def __init__(
        self,
        in_features: int,
        n_heads: int,
        fc_features: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(
            in_features=in_features, fc_features=fc_features, dropout=dropout
        )
        self.mhsa = MHSAModule(
            in_features=in_features, n_heads=n_heads, dropout=dropout
        )
        self.conv = ConformerConvModule(
            in_features=in_features, kernel_size=kernel_size, dropout=dropout
        )
        self.ff2 = FeedForwardModule(
            in_features=in_features, fc_features=fc_features, dropout=dropout
        )
        self.final_ln = nn.LayerNorm(in_features)

    def forward(
        self,
        x,
        key_padding_mask,
    ):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.final_ln(x)
        return x
