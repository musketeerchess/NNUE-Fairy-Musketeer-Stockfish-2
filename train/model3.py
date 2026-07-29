"""
Model 3 -- Betza-feature NNUE (Milestone 4).

Specification (contract, Milestone 4):
    "128 inputs in all cases (regardless of gating). Inputs: gating squares and
     additional features (number of controlled squares 1,2,3 and 4 square
     distance when the piece is in the middle of an empty board on e4;
     colour-boundness score ...; input infinite range like queen). Outputs 512
     neurons for all layers."

Architecture:

    input(128, MK-Model3 encoding)
      → Linear(128, 512) → clipped-ReLU
      → Linear(512, 512) → clipped-ReLU
      → Linear(512, 512) → clipped-ReLU
      → Linear(512, 1)

The 128 inputs come from ``features.encode_fen_model3``, which embeds the
Betza-derived geometry (controlled squares at distance 1..4, colour-boundness,
infinite-range) per piece type -- see that function's docstring.  The geometry
is validated in the Milestone-4 doc.

Usage:
    python train/model3.py --data data/processed/train1.jsonl --limit 40000 \
        --epochs 20 --lr 3e-3 --out models/model3_smoke.pt
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
from features import encode_fen_model3, N_FEATURES_M3   # noqa: E402

SCALE = 361.0
VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


class Model3(nn.Module):
    def __init__(self, width: int = 512, hidden: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(N_FEATURES_M3, width)]
            + [nn.Linear(width, width) for _ in range(hidden)])
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        for lin in self.layers:
            x = torch.clamp(lin(x), 0.0, 1.0)
        return self.output(x)


def load_dataset(patterns, limit):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no files matched {patterns}")
    key = os.path.join(os.path.dirname(files[0]),
                       f".cache3_{abs(hash((tuple(files), limit))) & 0xffffffff}.npz")
    if os.path.exists(key):
        d = np.load(key)
        print(f"loaded cache {key}: {len(d['X'])} rows")
        return d["X"], d["S"], d["R"]
    X, S, R = [], [], []
    t0 = time.time()
    for fn in files:
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("result_stm") is None:
                    continue
                X.append(encode_fen_model3(rec["fen"], VARIANT_MEN))
                S.append(rec["score_cp"])
                R.append((rec["result_stm"] + 1) / 2.0)
                if limit and len(X) >= limit:
                    break
        if limit and len(X) >= limit:
            break
    X = np.asarray(X, np.float32); S = np.asarray(S, np.float32); R = np.asarray(R, np.float32)
    print(f"encoded {len(X)} positions in {time.time()-t0:.1f}s")
    np.savez_compressed(key, X=X, S=S, R=R)
    return X, S, R


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, S, R = load_dataset(glob.glob(args.data) or [args.data], args.limit)
    X = torch.from_numpy(X); S = torch.from_numpy(S); R = torch.from_numpy(R)
    n = len(X); perm = torch.randperm(n); n_val = max(1, int(n * 0.05))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = Model3(width=args.width, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()):,}"
          f"  train={len(tr_idx)}  val={len(val_idx)}")

    def loss_fn(idx):
        x = X[idx].to(dev); s = S[idx].to(dev); r = R[idx].to(dev)
        q = model(x).squeeze(1)
        pred = torch.sigmoid(q / SCALE)
        target = lam * torch.sigmoid(s / SCALE) + (1 - lam) * r
        return ((pred - target) ** 2).mean()

    bs = args.batch
    for ep in range(1, args.epochs + 1):
        model.train()
        tp = tr_idx[torch.randperm(len(tr_idx))]
        run = 0.0; nb = 0
        for i in range(0, len(tp), bs):
            b = tp[i:i + bs]
            opt.zero_grad(); l = loss_fn(b); l.backward(); opt.step()
            run += l.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = loss_fn(val_idx).item()
        print(f"epoch {ep:2d}  train_loss {run/max(1,nb):.5f}  val_loss {vl:.5f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "width": args.width,
                "hidden": args.hidden, "scale": SCALE, "arch": "model3-betza"}, args.out)
    print(f"saved {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train1.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="models/model3.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
