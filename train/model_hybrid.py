"""
Hybrid NNUE: dense Betza-geometry features (Models 1-3 lineage) combined with
the sparse HalfKA king-relative accumulator, under one bucketed output head.

Both halves are letter-independent: the dense half is the identity-independent
geometry vector (``features.encode_fen_geo``), and the sparse half indexes by
canonical Betza id (``features_halfka``).  The idea is that the geometry half
supplies per-rule structure that generalises across armies, while the HalfKA
half supplies king-relative positional detail — the two signals NNUE usually
gets only from a huge feature set.

Output head is bucketed by gating phase (0/1/2), like the pure HalfKA net.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HybridNet(nn.Module):
    def __init__(self, dense_in: int, num_features: int, dim: int = 256,
                 geo_hidden: int = 64, buckets: int = 3, hidden: int = 32):
        super().__init__()
        self.dim = dim
        self.buckets = buckets
        self.ft = nn.EmbeddingBag(num_features, dim, mode="sum")     # sparse half
        self.dense = nn.Sequential(nn.Linear(dense_in, geo_hidden), nn.ReLU())
        head_in = 2 * dim + geo_hidden
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(head_in, hidden), nn.ReLU(),
                          nn.Linear(hidden, 1))
            for _ in range(buckets)
        ])

    def forward(self, dense, own_idx, own_off, opp_idx, opp_off, bucket):
        own = torch.clamp(self.ft(own_idx, own_off), 0.0, 1.0)
        opp = torch.clamp(self.ft(opp_idx, opp_off), 0.0, 1.0)
        d = self.dense(dense)
        x = torch.cat([own, opp, d], dim=1)
        out = x.new_zeros(x.size(0), 1)
        for b in range(self.buckets):
            m = (bucket == b)
            if m.any():
                out[m] = self.heads[b](x[m])
        return out
