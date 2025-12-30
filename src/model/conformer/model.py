import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer_blocks import ConformerBlocks
from .subsampling import Conv2dSubsampling


def lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return torch.arange(max_len, device=lengths.device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)


class ConformerCTCModel(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_mels=128,
        in_features=144,
        n_heads=4,
        fc_features=576,
        num_layers=4,
        kernel_size=31,
        dropout=0.1,
    ):
        super().__init__()
        self.subsampling = Conv2dSubsampling(
            n_mels=n_mels, in_features=in_features, dropout=dropout
        )

        self.encoder = ConformerBlocks(
            num_layers=num_layers,
            in_features=in_features,
            n_heads=n_heads,
            fc_features=fc_features,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.proj = nn.Linear(in_features, n_tokens)

    def forward(
        self, spectrogram: torch.Tensor, spectrogram_length: torch.Tensor, **batch
    ):
        x, out_len = self.subsampling(spectrogram, spectrogram_length)
        out_len = out_len.to(x.device)
        pad_mask = lengths_to_padding_mask(out_len, max_len=x.size(1))

        x = self.encoder(x, key_padding_mask=pad_mask)

        log_probs = F.log_softmax(self.proj(x), dim=-1)

        return {
            "log_probs": log_probs,
            "log_probs_length": out_len,
        }
