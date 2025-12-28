import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerCTCModel(nn.Module):
    def __init__(
        self,
        n_tokens: int,
        n_mels: int = 128,
        d_model: int = 144,
        **kwargs,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_mels = n_mels
        self.d_model = d_model
        self.proj = nn.Linear(d_model, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        pass
