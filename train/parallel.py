"""
Parallel feature-encoding pipeline for training on all CPU cores.

The bottleneck in every trainer here is the Python feature encoding
(`Board.from_fen` + building the feature vector), which is single-threaded.
This module moves that work into DataLoader worker processes, one per core, so
the machine's cores all encode at once while the main process does the (tiny)
optimiser step.  The model math is small; parallel encoding is where the speed
comes from.

Everything is top-level and picklable because Windows uses spawn: an
``IterableDataset`` streams the JSONL, shards lines across workers, chunk-shuffles
its shard, encodes, and yields samples; a per-architecture ``collate_*`` assembles
batches.  The registry, variants list, and the small spec dict are the only state
handed to the workers (all picklable); models never leave the main process.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features as F                                    # noqa: E402
import features_halfka as FK                            # noqa: E402
from musketeer import Board                             # noqa: E402

SCALE = 361.0
LAM = 0.7


def _target(rec):
    return LAM * (1.0 / (1.0 + np.exp(-rec["score_cp"] / SCALE))) + \
        (1 - LAM) * (rec["result_stm"] + 1) / 2.0


# --------------------------------------------------------------------------- #
# Per-worker encoder, built from a small picklable spec
# --------------------------------------------------------------------------- #
class _Encoder:
    """Encodes one record -> sample, configuring per-variant material as needed.
    Built fresh inside each worker so module globals are set per process."""

    def __init__(self, spec, reg):
        self.kind = spec["kind"]
        self.spec = spec
        self.reg = reg
        self.king_buckets = spec.get("king_buckets", 16)
        self.last_vm = None
        self.needs_material = self.kind in ("dense", "hybrid")
        if self.kind == "dense":
            enc = spec["encoding"]
            if enc == "geo":
                F.set_geo_registry(reg)
                self.fn = F.encode_fen_geo
            elif enc == "mk128":
                self.fn = F.encode_fen
            elif enc == "512":
                self.fn = F.encode_fen_512
            elif enc == "model3":
                F.set_model3_types(list("PNBRQKHU"))
                self.fn = F.encode_fen_model3
            else:
                raise ValueError(enc)
        elif self.kind in ("halfka", "halfkp"):
            self.include_kings = (self.kind == "halfka")
        elif self.kind == "hybrid":
            F.set_geo_registry(reg)
        else:
            raise ValueError(self.kind)

    def __call__(self, rec, vm):
        if self.needs_material and vm != self.last_vm:
            F.configure_piece_values(vm)
            self.last_vm = vm
        tgt = _target(rec)
        if self.kind == "dense":
            return (self.fn(rec["fen"], vm).astype(np.float32), tgt)
        if self.kind in ("halfka", "halfkp"):
            board = Board.from_fen(rec["fen"], vm)
            f = FK.board_features(board, self.reg, self.king_buckets,
                                  self.include_kings)
            return ((f["own"], f["opp"], f["stage"]), tgt)
        # hybrid
        board = Board.from_fen(rec["fen"], vm)
        dense = F.encode_board_geo(board, F._betza_map(vm), self.reg).astype(np.float32)
        f = FK.board_features(board, self.reg, self.king_buckets, True)
        return ((dense, f["own"], f["opp"], f["stage"]), tgt)


# --------------------------------------------------------------------------- #
# Iterable dataset: shard across workers, stride-sample, chunk-shuffle, encode
# --------------------------------------------------------------------------- #
class EncodingIterable(IterableDataset):
    def __init__(self, data, variants, reg, spec, val_keep, train_keep,
                 chunk_lines=250_000, seed=1, mode="train"):
        super().__init__()
        self.data = data
        self.variants = variants
        self.reg = reg
        self.spec = spec
        self.val_keep = val_keep
        self.train_keep = train_keep
        self.chunk_lines = chunk_lines
        self.seed = seed
        self.mode = mode                    # "train" (strided+shuffled) or "val"

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        enc = _Encoder(self.spec, self.reg)
        rng = random.Random(self.seed * 7919 + wid)
        buf = []

        def drain():
            rng.shuffle(buf)
            for r in buf:
                try:
                    s = enc(r, self.variants[r["vm"]])
                except Exception:
                    continue
                yield s
            buf.clear()

        is_val = (self.mode == "val")
        with open(self.data, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i % nw != wid:                       # this worker's shard
                    continue
                if is_val:
                    if i % self.val_keep != 0:          # validation: the held-out lines
                        continue
                else:
                    if i % self.val_keep == 0:          # training: exclude validation
                        continue
                    if i % self.train_keep != 0:        # ... stride-sampled across file
                        continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("result_stm") is None:
                    continue
                buf.append(rec)
                if len(buf) >= self.chunk_lines:
                    yield from drain()
            if buf:
                yield from drain()


# --------------------------------------------------------------------------- #
# Collate functions (one per architecture kind)
# --------------------------------------------------------------------------- #
def collate_dense(samples):
    X = np.asarray([s[0] for s in samples], dtype=np.float32)
    S = np.asarray([s[1] for s in samples], dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(S)


def _bag(index_lists):
    idx, off = [], []
    for lst in index_lists:
        off.append(len(idx))
        idx.extend(lst)
    return (torch.tensor(idx, dtype=torch.long),
            torch.tensor(off, dtype=torch.long))


def collate_halfka(samples):
    own_i, own_o = _bag([s[0][0] for s in samples])
    opp_i, opp_o = _bag([s[0][1] for s in samples])
    bk = torch.tensor([s[0][2] for s in samples], dtype=torch.long)
    S = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    return (own_i, own_o, opp_i, opp_o, bk), S


def collate_hybrid(samples):
    dense = torch.from_numpy(np.asarray([s[0][0] for s in samples], np.float32))
    own_i, own_o = _bag([s[0][1] for s in samples])
    opp_i, opp_o = _bag([s[0][2] for s in samples])
    bk = torch.tensor([s[0][3] for s in samples], dtype=torch.long)
    S = torch.tensor([s[1] for s in samples], dtype=torch.float32)
    return (dense, own_i, own_o, opp_i, opp_o, bk), S


COLLATE = {"dense": collate_dense, "halfka": collate_halfka,
           "halfkp": collate_halfka, "hybrid": collate_hybrid}
