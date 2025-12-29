import torch
import torch.nn as nn


class Conv2dSubsampling(nn.Module):
    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        f = (n_mels + 1) // 2
        f = (f + 1) // 2
        self.proj = nn.Linear(d_model * f, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, spec: torch.Tensor, lengths: torch.Tensor):
        x = self.conv(spec.unsqueeze(1))
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = self.drop(self.proj(x))

        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        return x, lengths
