"""
Model 2 -- the gating-adaptive Musketeer NNUE (Milestone 3).

Specification (contract, Milestone 3):
    "80 or 128 inputs till gating is finished, then switch back to 64 inputs.
     The number of neurons is 512 in case we use 80/128 inputs and 256 in case
     we use 64 inputs. The model should adapt the number of neurons related to
     gating."
    (+ earlier: "subsequent layers match the neuron count of the first layer.")

So Model 2 is *two* sub-networks and every position is routed to one of them by
its gating state:

    gating NOT finished  ->  input(128) → 512 → 512 → 1     ("pre-gating" path)
    gating finished      ->  input( 64) → 256 → 256 → 1     ("post-gating" path)

The routing uses ``features.gating_finished``.  The 64-input vector is just the
board plane of MK128 (its gating plane is already all-zero once gating is done),
so both paths share the same encoder -- no separate feature code.

Usage (CPU smoke test):
    python train/model2.py --data data/processed/train1.jsonl --limit 40000 \
        --epochs 20 --lr 3e-3 --out models/model2_smoke.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import encode_fen, N_FEATURES            # noqa: E402

SCALE = 361.0
VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class _Path(nn.Module):
    """One regime's sub-network: input → width → width(×hidden) → 1."""
    def __init__(self, n_in: int, width: int, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(n_in, width)] + [nn.Linear(width, width) for _ in range(hidden)])
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        for lin in self.layers:
            x = torch.clamp(lin(x), 0.0, 1.0)
        return self.output(x)


class Model2(nn.Module):
    def __init__(self, hidden: int = 1):
        super().__init__()
        self.pre = _Path(128, 512, hidden)     # gating unfinished
        self.post = _Path(64, 256, hidden)     # gating finished

    def forward_pre(self, x128):
        return self.pre(x128)

    def forward_post(self, x64):
        return self.post(x64)


# --------------------------------------------------------------------------- #
# Data  (also records the per-position gating regime)
# --------------------------------------------------------------------------- #
def fen_gating_finished(fen: str) -> bool:
    rows = fen.split()[0].split("/")
    return set(rows[0]) <= {"*"} and set(rows[9]) <= {"*"}


def load_dataset(patterns, limit):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no files matched {patterns}")
    key = os.path.join(os.path.dirname(files[0]),
                       f".cache2_{abs(hash((tuple(files), limit))) & 0xffffffff}.npz")
    if os.path.exists(key):
        d = np.load(key)
        print(f"loaded cache {key}: {len(d['X'])} rows")
        return d["X"], d["S"], d["R"], d["G"]

    X, S, R, G = [], [], [], []
    t0 = time.time()
    for fn in files:
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("result_stm") is None:
                    continue
                X.append(encode_fen(rec["fen"], VARIANT_MEN))
                S.append(rec["score_cp"])
                R.append((rec["result_stm"] + 1) / 2.0)
                G.append(1 if fen_gating_finished(rec["fen"]) else 0)
                if limit and len(X) >= limit:
                    break
        if limit and len(X) >= limit:
            break
    X = np.asarray(X, np.float32); S = np.asarray(S, np.float32)
    R = np.asarray(R, np.float32); G = np.asarray(G, np.int64)
    print(f"encoded {len(X)} positions in {time.time()-t0:.1f}s "
          f"(post-gating {int(G.sum())}, pre-gating {int((G==0).sum())})")
    np.savez_compressed(key, X=X, S=S, R=R, G=G)
    return X, S, R, G


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, S, R, G = load_dataset(glob.glob(args.data) or [args.data], args.limit)
    X = torch.from_numpy(X); S = torch.from_numpy(S)
    R = torch.from_numpy(R); G = torch.from_numpy(G)
    n = len(X)
    perm = torch.randperm(n)
    n_val = max(1, int(n * 0.05))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = Model2(hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()):,}"
          f"  train={len(tr_idx)}  val={len(val_idx)}")

    def batch_loss(idx):
        g = G[idx]
        pre_i = idx[g == 0]
        post_i = idx[g == 1]
        total = torch.zeros((), device=dev)
        cnt = 0
        for sub, is_post in ((pre_i, False), (post_i, True)):
            if len(sub) == 0:
                continue
            x = X[sub].to(dev)
            s = S[sub].to(dev); r = R[sub].to(dev)
            q = (model.forward_post(x[:, :64]) if is_post
                 else model.forward_pre(x)).squeeze(1)
            pred = torch.sigmoid(q / SCALE)
            target = lam * torch.sigmoid(s / SCALE) + (1 - lam) * r
            total = total + ((pred - target) ** 2).sum()
            cnt += len(sub)
        return total / max(1, cnt)

    bs = args.batch
    for ep in range(1, args.epochs + 1):
        model.train()
        tp = tr_idx[torch.randperm(len(tr_idx))]
        running = 0.0; nb = 0
        for i in range(0, len(tp), bs):
            b = tp[i:i + bs]
            opt.zero_grad()
            l = batch_loss(b)
            l.backward(); opt.step()
            running += l.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = batch_loss(val_idx).item()
        print(f"epoch {ep:2d}  train_loss {running/max(1,nb):.5f}  val_loss {vl:.5f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden": args.hidden,
                "scale": SCALE, "arch": "model2-gating-adaptive"}, args.out)
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train1.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=1)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="models/model2.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
