"""
Head-to-head comparison of every Musketeer NNUE architecture on the big database.

Trains each model on the SAME held-out split with the SAME position budget, so
the validation losses are directly comparable, and prints a ranked table.  This
is the fair "compare Models 1-4 vs geometry vs HalfKP/HalfKA vs hybrid" pass the
client asked for.  The winner can then be retrained on the full dataset (ideally
on a GPU) with its own trainer.

Every model here is letter-independent: the dense geometry encoder and the
HalfKA feature set both key on the canonical Betza id (see betza_id).

Split policy (deterministic, streaming, no split file needed):
    line i is VALIDATION iff  i % val_every == 0  and  i < val_cap*val_every
    every other line is TRAINING
so validation is a fixed, held-out sample disjoint from training.

Resumable: results are appended to models/compare_results.json after each model,
and models already present are skipped, so an interrupted run continues.

Usage:
    python train/compare_models.py --data data/bigdb/bigdb.jsonl \
        --variants data/bigdb/variants.json \
        --train-positions 3000000 --val-cap 50000
    python train/compare_models.py ... --only geo,halfka,hybrid   # subset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import betza_id as BID                                  # noqa: E402
import features as F                                    # noqa: E402
import features_halfka as FK                            # noqa: E402
from musketeer import Board                             # noqa: E402
from model1 import Model1                               # noqa: E402
from model2 import Model2                               # noqa: E402
from model3 import Model3                               # noqa: E402
from model4 import Model4                               # noqa: E402
from model512 import Model512                           # noqa: E402
from model_halfka import HalfKANet                      # noqa: E402
from model_hybrid import HybridNet                      # noqa: E402

SCALE = 361.0
LAM = 0.7


def _keep_awake():
    """Ask Windows not to sleep while this run is in progress (reverts on exit;
    not a persistent power-setting change).  No-op on other platforms."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("keep-awake enabled (system will not idle-sleep during the run)",
              flush=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Streaming split
# --------------------------------------------------------------------------- #
def iter_split(path, role_of, roles):
    """Yield (role, rec) for lines whose role is in ``roles``."""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            role = role_of(i)
            if role not in roles:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("result_stm") is None:
                continue
            yield role, rec


def make_role_of(total_lines, val_cap, budget, epochs):
    """Stride the split across the WHOLE file so both validation and the
    budget-limited training sample span every variant, not just the first ones
    (the file is ordered by parse order: old armies first, new armies appended).
    """
    val_keep = max(1, total_lines // max(1, val_cap))
    need = max(1, budget * max(1, epochs))
    train_keep = max(1, total_lines // need)

    def role_of(i):
        if i % val_keep == 0:            # validation: ~val_cap, spread across file
            return "val"
        if i % train_keep == 0:          # training: ~budget*epochs, spread across file
            return "train"
        return "skip"
    return role_of, val_keep, train_keep


def train_records(data, role_of, variants, chunk_lines, seed):
    """Chunk-shuffled stream of (rec, vm) training records."""
    rng = random.Random(seed)
    buf = []
    for _, rec in iter_split(data, role_of, ("train",)):
        buf.append(rec)
        if len(buf) >= chunk_lines:
            rng.shuffle(buf)
            for r in buf:
                yield (r, variants[r["vm"]])
            buf = []
    if buf:
        rng.shuffle(buf)
        for r in buf:
            yield (r, variants[r["vm"]])


def target(rec):
    """WDL+eval blend target used by every trainer here."""
    return LAM * _sig(rec["score_cp"]) + (1 - LAM) * (rec["result_stm"] + 1) / 2.0


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x / SCALE))


# --------------------------------------------------------------------------- #
# Per-architecture strategies: build model, encode a record, collate, forward
# --------------------------------------------------------------------------- #
class DenseStrategy:
    """mk128 / 512 / model3 / geo — a dense vector into a plain MLP."""

    def __init__(self, name, encoding, arch, reg):
        self.name = name
        self.encoding = encoding
        self.arch = arch
        self.reg = reg
        if encoding == "geo":
            F.set_geo_registry(reg)
            self.encode_fn, self.n_in = F.encode_fen_geo, F.n_features_geo(reg)
        elif encoding == "mk128":
            self.encode_fn, self.n_in = F.encode_fen, F.N_FEATURES
        elif encoding == "512":
            self.encode_fn, self.n_in = F.encode_fen_512, F.N_FEATURES_512
        elif encoding == "model3":
            F.set_model3_types(list("PNBRQKHU"))
            self.encode_fn, self.n_in = F.encode_fen_model3, F.N_FEATURES_M3
        else:
            raise ValueError(encoding)

    def build(self, dev):
        a, n = self.arch, self.n_in
        if a == "model1":
            m = Model1(n_in=n, width=256, hidden=2)
        elif a == "model2":
            m = Model2(hidden=1)
        elif a == "model3":
            m = Model3(width=512, hidden=2)
        elif a == "model4":
            m = Model4(width=256, hidden=4)
        elif a == "model512":
            m = Model512(n_in=n, width=512, hidden=3)
        else:
            raise ValueError(a)
        return m.to(dev)

    def encode(self, rec, vm):
        return self.encode_fn(rec["fen"], vm)

    def collate(self, samples, dev):
        X = torch.from_numpy(np.asarray([s[0] for s in samples], np.float32)).to(dev)
        S = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=dev)
        return (X,), S

    def forward(self, model, inp):
        x = inp[0]
        if self.arch == "model2":
            # route by gating state: mk128 gating plane (64:128) is all-zero
            # once gating is finished -> post-gating 64-input path, else pre.
            post = x[:, 64:128].abs().sum(1) == 0
            q = x.new_zeros(x.size(0), 1)
            if (~post).any():
                q[~post] = model.forward_pre(x[~post])
            if post.any():
                q[post] = model.forward_post(x[post][:, :64])
            return q.squeeze(1)
        return model(x).squeeze(1)


class HalfKAStrategy:
    def __init__(self, name, mode, reg, king_buckets, dim, hidden):
        self.name = name
        self.include_kings = (mode == "halfka")
        self.reg = reg
        self.king_buckets = king_buckets
        self.dim = dim
        self.hidden = hidden
        self.n_feat = FK.num_features(reg, king_buckets)

    def build(self, dev):
        return HalfKANet(self.n_feat, dim=self.dim, buckets=3,
                         hidden=self.hidden).to(dev)

    def encode(self, rec, vm):
        board = Board.from_fen(rec["fen"], vm)
        f = FK.board_features(board, self.reg, self.king_buckets,
                              self.include_kings)
        return (f["own"], f["opp"], f["stage"])

    def collate(self, samples, dev):
        oi, oo, pi, po, bk, S = [], [], [], [], [], []
        for (own, opp, stage), tgt in ((s[0], s[1]) for s in samples):
            oo.append(len(oi)); oi.extend(own)
            po.append(len(pi)); pi.extend(opp)
            bk.append(stage); S.append(tgt)
        t = lambda a, dt: torch.tensor(a, dtype=dt, device=dev)          # noqa: E731
        return ((t(oi, torch.long), t(oo, torch.long),
                 t(pi, torch.long), t(po, torch.long), t(bk, torch.long)),
                t(S, torch.float32))

    def forward(self, model, inp):
        return model(*inp).squeeze(1)


class HybridStrategy:
    def __init__(self, name, reg, king_buckets, dim, geo_hidden, hidden):
        self.name = name
        self.reg = reg
        self.king_buckets = king_buckets
        self.dim = dim
        self.geo_hidden = geo_hidden
        self.hidden = hidden
        F.set_geo_registry(reg)
        self.dense_in = F.n_features_geo(reg)
        self.n_feat = FK.num_features(reg, king_buckets)

    def build(self, dev):
        return HybridNet(self.dense_in, self.n_feat, dim=self.dim,
                         geo_hidden=self.geo_hidden, buckets=3,
                         hidden=self.hidden).to(dev)

    def encode(self, rec, vm):
        board = Board.from_fen(rec["fen"], vm)
        dense = F.encode_board_geo(board, F._betza_map(vm), self.reg)
        f = FK.board_features(board, self.reg, self.king_buckets, True)
        return (dense, f["own"], f["opp"], f["stage"])

    def collate(self, samples, dev):
        dense = torch.from_numpy(
            np.asarray([s[0][0] for s in samples], np.float32)).to(dev)
        oi, oo, pi, po, bk, S = [], [], [], [], [], []
        for s in samples:
            (_, own, opp, stage), tgt = s[0], s[1]
            oo.append(len(oi)); oi.extend(own)
            po.append(len(pi)); pi.extend(opp)
            bk.append(stage); S.append(tgt)
        t = lambda a, dt: torch.tensor(a, dtype=dt, device=dev)          # noqa: E731
        return ((dense, t(oi, torch.long), t(oo, torch.long),
                 t(pi, torch.long), t(po, torch.long), t(bk, torch.long)),
                t(S, torch.float32))

    def forward(self, model, inp):
        return model(*inp).squeeze(1)


# --------------------------------------------------------------------------- #
# Generic train / eval loop over a strategy
# --------------------------------------------------------------------------- #
def _batches(records, strat, batch, dev, configure_material):
    """Encode a stream of (rec, vm) into collated batches; configure cp material
    per variant for the dense encoders that use it."""
    buf, last_vm = [], None
    for rec, vm in records:
        if configure_material and vm != last_vm:
            F.configure_piece_values(vm)
            last_vm = vm
        try:
            enc = strat.encode(rec, vm)
        except Exception:
            continue
        buf.append((enc, target(rec)))
        if len(buf) >= batch:
            yield strat.collate(buf, dev)
            buf = []
    if buf:
        yield strat.collate(buf, dev)


def evaluate(strat, model, val_records, batch, dev, configure_material):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for inp, S in _batches(val_records, strat, batch, dev, configure_material):
            q = strat.forward(model, inp)
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            tot += loss.item() * S.numel(); n += S.numel()
    return tot / max(1, n)


def run_one(strat, data, variants, role_of, val_recs, budget, epochs,
            batch, lr, dev, configure_material, log_every, chunk_lines):
    model = strat.build(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {strat.name} | params {params:,} | budget {budget:,} pos ===",
          flush=True)
    t0 = time.time(); seen = 0; done = False
    for ep in range(1, epochs + 1):
        if done:
            break
        model.train()
        train_stream = train_records(data, role_of, variants, chunk_lines, seed=ep)
        for inp, S in _batches(train_stream, strat, batch, dev, configure_material):
            opt.zero_grad()
            q = strat.forward(model, inp)
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward(); opt.step()
            seen += S.numel()
            if seen // batch % log_every == 0:
                el = time.time() - t0
                print(f"  {strat.name}: {seen:,} pos  {seen/el:.0f} pos/s  "
                      f"train {loss.item():.5f}  {el/60:.1f} min", flush=True)
            if seen >= budget:
                done = True
                break
    val_records = ((r, variants[r["vm"]]) for r in val_recs)
    vloss = evaluate(strat, model, val_records, batch, dev, configure_material)
    dt = (time.time() - t0) / 60
    print(f"--- {strat.name}: VAL {vloss:.5f}  ({seen:,} pos, {dt:.1f} min, "
          f"{params:,} params) ---", flush=True)
    return {"name": strat.name, "val_loss": round(vloss, 5), "params": params,
            "train_pos": seen, "minutes": round(dt, 1)}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--train-positions", type=int, default=3_000_000,
                    help="position budget PER model (equal footing)")
    ap.add_argument("--epochs", type=int, default=1,
                    help="max passes over train data (budget usually hits first)")
    ap.add_argument("--val-cap", type=int, default=50_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914,
                    help="line count of --data, used to stride the split")
    ap.add_argument("--chunk-lines", type=int, default=1_000_000,
                    help="shuffle-buffer size for training")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--king-buckets", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--only", default="",
                    help="comma list to restrict, e.g. geo,halfka,hybrid")
    ap.add_argument("--out", default="models/compare_results.json")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    _keep_awake()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    role_of, val_keep, train_keep = make_role_of(
        args.total_lines, args.val_cap, args.train_positions, args.epochs)

    print(f"device={dev}  canonical_types={reg.num_types}  "
          f"budget/model={args.train_positions:,}  "
          f"val_stride={val_keep}  train_stride={train_keep}", flush=True)
    print("collecting held-out validation set (spans all variants) ...", flush=True)
    val_recs = [rec for role, rec in iter_split(args.data, role_of, ("val",))]
    nvar = len({r["vm"] for r in val_recs})
    print(f"validation positions: {len(val_recs):,}  covering {nvar} variants",
          flush=True)

    # every architecture, and whether its encoder needs cp material configured
    catalog = {
        "model1_mk128": (DenseStrategy("model1_mk128", "mk128", "model1", reg), True),
        "model2_mk128": (DenseStrategy("model2_mk128", "mk128", "model2", reg), True),
        "model3_legacy": (DenseStrategy("model3_legacy", "model3", "model3", reg), True),
        "model4_mk128": (DenseStrategy("model4_mk128", "mk128", "model4", reg), True),
        "geo": (DenseStrategy("geo", "geo", "model1", reg), True),
        "net512": (DenseStrategy("net512", "512", "model512", reg), True),
        "halfkp": (HalfKAStrategy("halfkp", "halfkp", reg, args.king_buckets,
                                  args.dim, 32), False),
        "halfka": (HalfKAStrategy("halfka", "halfka", reg, args.king_buckets,
                                  args.dim, 32), False),
        "hybrid": (HybridStrategy("hybrid", reg, args.king_buckets, args.dim,
                                  64, 32), True),
    }
    order = ["model1_mk128", "model2_mk128", "model3_legacy", "model4_mk128",
             "geo", "net512", "halfkp", "halfka", "hybrid"]
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        order = [k for k in order if k in want]

    # resume: load prior results, skip finished models
    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    done_names = {r["name"] for r in results}

    for key in order:
        strat, configure_material = catalog[key]
        if strat.name in done_names:
            print(f"skip {strat.name} (already in {args.out})", flush=True)
            continue
        try:
            res = run_one(strat, args.data, variants, role_of, val_recs,
                          args.train_positions, args.epochs, args.batch,
                          args.lr, dev, configure_material, args.log_every,
                          args.chunk_lines)
        except Exception as e:
            res = {"name": strat.name, "error": f"{type(e).__name__}: {e}"}
            print(f"!! {strat.name} failed: {res['error']}", flush=True)
        results.append(res)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=2)

    # final ranked table
    ranked = sorted((r for r in results if "val_loss" in r),
                    key=lambda r: r["val_loss"])
    print("\n" + "=" * 64)
    print(f"{'model':<16}{'val_loss':>10}{'params':>14}{'min':>8}")
    print("-" * 64)
    for r in ranked:
        print(f"{r['name']:<16}{r['val_loss']:>10.5f}{r['params']:>14,}"
              f"{r['minutes']:>8.1f}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<16}   ERROR: {r['error']}")
    print("=" * 64)
    print(f"results saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
