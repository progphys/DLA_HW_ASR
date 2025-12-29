import torch
import torch.nn as nn
import torch.nn.functional as F
from ffn import Swish


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pointwise_in = nn.Conv1d(
            in_channels=in_features,
            out_channels=2 * in_features,
            kernel_size=1,
            bias=True,
        )

        self.depthwise = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_features,
            bias=True,
        )

        self.bn = nn.BatchNorm1d(in_features)
        self.act = Swish()

        self.pointwise_out = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=1,
            bias=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, T, C)->(B, C, T)

        x = self.pointwise_in(x)  # (B, 2C, T)
        x = F.glu(x, dim=1)  # (B, C, T)

        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (B, C, T) -> (B, T, C)


if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, C = 2, 120, 144

    module = ConformerConvModule(
        in_features=C,
        kernel_size=31,
        dropout=0.1,
    )

    x = torch.randn(B, T, C, requires_grad=True)
    y = module(x)

    print("input shape :", x.shape)
    print("output shape:", y.shape)
    assert y.shape == x.shape

    loss = y.mean()
    loss.backward()

    print("grad shape :", x.grad.shape)
