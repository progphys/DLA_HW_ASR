import math
import re
from string import ascii_lowercase

import torch

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        if isinstance(inds, torch.Tensor):
            inds = inds.detach().cpu()
        elif not torch.is_tensor(inds):
            inds = torch.as_tensor(inds)
        # delete blacnk symbols
        inds = inds[inds != 0]

        if inds.numel() == 0:
            return ""

        keep = torch.ones_like(inds, dtype=torch.bool)
        # delete repeat symbols
        keep[1:] = inds[1:] != inds[:-1]
        inds = inds[keep]

        return "".join(self.ind2char[int(i)] for i in inds)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text

    @staticmethod
    def lse(a, b):
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        m = a if a > b else b
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    def ctc_beam_search(self, log_probs, beam_size=10):
        blank = self.char2ind[self.EMPTY_TOK]
        lp = log_probs.detach().cpu()

        beams = {(): (0.0, -math.inf)}

        for t in range(lp.size(0)):
            new = {}

            top = sorted(
                beams.items(),
                key=lambda kv: self.lse(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_size]

            for pref, (pb, pnb) in top:
                last = pref[-1] if pref else None

                for c in range(lp.size(1)):
                    p = float(lp[t, c])

                    if c == blank:
                        nb, nnb = new.get(pref, (-math.inf, -math.inf))
                        nb = self.lse(nb, pb + p)
                        nb = self.lse(nb, pnb + p)
                        new[pref] = (nb, nnb)
                    else:
                        if last == c:
                            pref2 = pref + (c,)
                            nb, nnb = new.get(pref2, (-math.inf, -math.inf))
                            nnb = self.lse(nnb, pb + p)
                            new[pref2] = (nb, nnb)

                            sb, snb = new.get(pref, (-math.inf, -math.inf))
                            snb = self.lse(snb, pnb + p)
                            new[pref] = (sb, snb)
                        else:
                            pref2 = pref + (c,)
                            nb, nnb = new.get(pref2, (-math.inf, -math.inf))
                            nnb = self.lse(nnb, pb + p)
                            nnb = self.lse(nnb, pnb + p)
                            new[pref2] = (nb, nnb)

            beams = new

        best = max(beams.items(), key=lambda kv: self.lse(kv[1][0], kv[1][1]))[0]
        return self.ctc_decode(torch.tensor(best))
