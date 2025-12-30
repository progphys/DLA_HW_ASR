import random

import torchaudio
from torch import nn


class FreqMask(nn.Module):
    def __init__(self, p=0.8, freq_mask_param=10, iid_masks=False):
        super().__init__()
        self.p = p
        self.t = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
        )

    def forward(self, x):
        if self.p <= 0 or random.random() > self.p:
            return x
        return self.t(x)
