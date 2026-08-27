"""
Activations for the NNUE accumulator.

Standard NNUE clamps the feature-transformer output to [0, 1] (clipped ReLU),
which is what an int8/int16 quantized engine wants. That clamp throws away the
"negative" side of the accumulator — in chess terms, information that the
position favours the opponent. Zied's idea is to let the network choose its own
clipping interval, asymmetric and learned, so it can keep some of that negative
range if the data says it helps.

- ClippedReLU:          fixed [lo, hi], default [0, 1] (the standard baseline).
- LearnableClippedReLU: [lo, lo + width] with lo and width learned. Written with
  minimum/maximum so gradients flow to the bounds in the saturated region, and
  parameterised through a positive width so the interval can never invert.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        super().__init__()
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, x):
        return torch.clamp(x, self.lo, self.hi)

    def bounds(self):
        return (self.lo, self.hi)


class LearnableClippedReLU(nn.Module):
    """Asymmetric clipped ReLU with learnable lower bound and width."""

    def __init__(self, lo_init: float = -1.0, hi_init: float = 4.0):
        super().__init__()
        self.lo = nn.Parameter(torch.tensor(float(lo_init)))
        # width = hi - lo kept strictly positive via exp(log_width)
        self.log_width = nn.Parameter(torch.tensor(float(hi_init - lo_init)).log())

    def forward(self, x):
        hi = self.lo + torch.exp(self.log_width)
        return torch.minimum(torch.maximum(x, self.lo), hi)

    def bounds(self):
        with torch.no_grad():
            lo = self.lo.item()
            return (lo, lo + float(torch.exp(self.log_width)))


def make_activation(kind: str, lo_init: float = -1.0, hi_init: float = 4.0):
    """kind: 'clip' -> fixed [0,1]; 'learn' -> learnable asymmetric [lo, hi]."""
    if kind == "clip":
        return ClippedReLU(0.0, 1.0)
    if kind == "learn":
        return LearnableClippedReLU(lo_init, hi_init)
    raise ValueError(f"unknown activation {kind}")
