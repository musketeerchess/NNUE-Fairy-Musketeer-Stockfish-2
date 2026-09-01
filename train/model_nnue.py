"""
One configurable NNUE, so a single class covers every architecture Zied listed.

Shape (standard NNUE): a feature transformer (EmbeddingBag) of width `dim` sums
the active king-relative features of each perspective into an accumulator; the
two accumulators (own, opponent) are clamped by the chosen activation and
concatenated, an optional dense branch (Betza geometry, or gating+Betza for the
Ultra model) is concatenated too, and a gating-phase-bucketed head reads out the
evaluation.

`dim` is the per-perspective transformer width, so `dim`x2 is the concatenated
accumulator, matching the "256x2 -> 64 -> 32 -> 1" notation. `head` is the tuple
of hidden widths after the accumulator, e.g. (64, 32) or (8, 32).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from activations import make_activation


class NNUENet(nn.Module):
    def __init__(self, num_features: int, dim: int = 256, head=(64, 32),
                 buckets: int = 3, dense_in: int = 0, act: str = "clip",
                 lo_init: float = -1.0, hi_init: float = 4.0, sparse: bool = True):
        super().__init__()
        self.dim = dim
        self.buckets = buckets
        self.dense_in = dense_in
        self.act_kind = act
        self.head_cfg = tuple(head)
        self.sparse = sparse
        # sparse gradients update only the rows a batch touches, which keeps a
        # large feature transformer trainable on CPU (SparseAdam). On a GPU the
        # dense form (sparse=False, plain Adam) is simple and fits in VRAM.
        self.ft = nn.EmbeddingBag(num_features, dim, mode="sum", sparse=sparse)
        # one activation instance per accumulator use; learnable variant shares params
        self.act = make_activation(act, lo_init, hi_init)
        if dense_in:
            self.dense = nn.Sequential(nn.Linear(dense_in, dim), nn.ReLU())
        head_in = 2 * dim + (dim if dense_in else 0)

        def make_head():
            layers, prev = [], head_in
            for h in head:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, 1)]
            return nn.Sequential(*layers)

        self.heads = nn.ModuleList([make_head() for _ in range(buckets)])

    def forward(self, own_idx, own_off, opp_idx, opp_off, bucket, dense=None):
        own = self.act(self.ft(own_idx, own_off))
        opp = self.act(self.ft(opp_idx, opp_off))
        parts = [own, opp]
        if self.dense_in and dense is not None:
            parts.append(self.dense(dense))
        x = torch.cat(parts, dim=1)
        out = x.new_zeros(x.size(0), 1)
        for b in range(self.buckets):
            m = (bucket == b)
            if m.any():
                out[m] = self.heads[b](x[m])
        return out

    # ---- optimizer param groups ------------------------------------------ #
    def param_groups(self):
        """Return (sparse feature-transformer params, all other params) so the
        caller can drive the FT with SparseAdam and the rest with Adam."""
        ft = list(self.ft.parameters())
        ft_ids = {id(p) for p in ft}
        other = [p for p in self.parameters() if id(p) not in ft_ids]
        return ft, other

    # ---- reporting ------------------------------------------------------- #
    def activation_bounds(self):
        return self.act.bounds()

    def nnue_size(self):
        """Parameter count and an estimated quantized .nnue size in bytes.

        Following the engine convention, the feature-transformer weights are
        stored as int16 (2 bytes) and the small dense layers as int8 (1 byte).
        The transformer dominates, so this is close to what a real .nnue would
        weigh on disk."""
        ft_params = sum(p.numel() for p in self.ft.parameters())
        other = sum(p.numel() for p in self.parameters()) - ft_params
        if self.dense_in:
            other += sum(p.numel() for p in self.dense.parameters()) * 0  # counted in other
        bytes_est = ft_params * 2 + other * 1
        total = ft_params + other
        return {"params": total, "ft_params": ft_params,
                "bytes": bytes_est, "mb": round(bytes_est / 1e6, 2)}
