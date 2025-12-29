import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        in_features: int,
        fc_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, fc_features)
        self.act = Swish()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_features, in_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C = 2, 100, 144
    FF = 576
    module = FeedForwardModule(
        in_features=C,
        fc_features=FF,
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
