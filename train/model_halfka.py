"""
HalfKP / HalfKA NNUE network for Musketeer.

Standard NNUE shape: a shared feature transformer (an EmbeddingBag that sums the
active king-relative features of a perspective into a dense accumulator), applied
to both the side-to-move and the opponent perspective, concatenated, then a small
output head.  The head is **bucketed by gating phase** (0/1/2 = how many sides
have finished gating), which is the client's three-stage split.

The feature-transformer input dimension is ``features_halfka.num_features(reg,
king_buckets)``; the "piece type" part of every feature is a canonical Betza id,
so the whole network is letter-independent by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HalfKANet(nn.Module):
    def __init__(self, num_features: int, dim: int = 256,
                 buckets: int = 3, hidden: int = 32):
        super().__init__()
        self.dim = dim
        self.buckets = buckets
        # sum active features per perspective into a dim-wide accumulator
        self.ft = nn.EmbeddingBag(num_features, dim, mode="sum")
        # one output head per gating phase; input is both accumulators concatenated
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
            for _ in range(buckets)
        ])

    def forward(self, own_idx, own_off, opp_idx, opp_off, bucket):
        # clipped-ReLU accumulators, exactly like a quantised NNUE
        own = torch.clamp(self.ft(own_idx, own_off), 0.0, 1.0)
        opp = torch.clamp(self.ft(opp_idx, opp_off), 0.0, 1.0)
        x = torch.cat([own, opp], dim=1)
        out = x.new_zeros(x.size(0), 1)
        for b in range(self.buckets):
            m = (bucket == b)
            if m.any():
                out[m] = self.heads[b](x[m])
        return out
