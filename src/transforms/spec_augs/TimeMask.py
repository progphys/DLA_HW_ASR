import random

import torchaudio
from torch import nn


class TimeMask(nn.Module):
    def __init__(self, p=0.8, time_mask_param=30, iid_masks=False):
        super().__init__()
        self.p = p
        self.t = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param,
            iid_masks=iid_masks,
        )

    def forward(self, x):
        if self.p <= 0 or random.random() > self.p:
            return x
        return self.t(x)
