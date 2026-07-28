"""
Model 1 -- the first Musketeer NNUE (Milestone 2).

Specification (client's contract, Milestone 2):
    "First layer must have ... 128 inputs, these inputs should be transformed
     with 256 neurons.  Then next hidden layers will use 256 neurons."

So Model 1 is a fully-connected evaluation network:

    input(128)  ->  FT Linear(128, 256)  -> clipped-ReLU
                ->  Linear(256, 256)      -> clipped-ReLU
                ->  Linear(256, 256)      -> clipped-ReLU
                ->  Linear(256, 1)        =  evaluation (side-to-move cp)

Input features: the MK128 encoding from ``src/features.py``.

Training objective is the standard NNUE loss: the network's evaluation is
squashed to a win probability and regressed against a blend of the engine's
own evaluation (from the PGN) and the actual game result:

    target = lambda * sigmoid(score/scale) + (1-lambda) * wdl_result
    loss   = MSE( sigmoid(eval/scale), target )

Usage (CPU smoke test):
    python train/model1.py --data data/processed/train1.jsonl --limit 20000 \
        --epochs 3 --out models/model1_smoke.pt
Full run (GPU recommended):
    python train/model1.py --data "data/processed/uni_hawk_training_nnue-*.jsonl" \
        --epochs 30 --out models/model1.pt
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
from features import encode_fen, N_FEATURES          # noqa: E402

SCALE = 361.0            # cp -> win-prob scaling (Stockfish-style)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Model1(nn.Module):
    def __init__(self, n_in: int = N_FEATURES, width: int = 256, hidden: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(n_in, width)]      # feature transformer
        for _ in range(hidden):
            layers.append(nn.Linear(width, width))              # hidden 256 layers
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = torch.clamp(lin(x), 0.0, 1.0)                   # clipped ReLU (NNUE)
        return self.output(x)                                   # eval in cp-ish units


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_dataset(patterns: list[str], limit: int | None, cache: bool = True):
    files: list[str] = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no files matched {patterns}")

    key = os.path.join(
        os.path.dirname(files[0]),
        f".cache_{abs(hash((tuple(files), limit))) & 0xffffffff}.npz")
    if cache and os.path.exists(key):
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
                # VariantMen is fixed for this dataset; embed it once.
                X.append(encode_fen(rec["fen"], VARIANT_MEN))
                S.append(rec["score_cp"])
                R.append((rec["result_stm"] + 1) / 2.0)         # {-1,0,1}->{0,.5,1}
                if limit and len(X) >= limit:
                    break
        if limit and len(X) >= limit:
            break
    X = np.asarray(X, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32)
    print(f"encoded {len(X)} positions in {time.time()-t0:.1f}s")
    if cache:
        np.savez_compressed(key, X=X, S=S, R=R)
    return X, S, R


VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, S, R = load_dataset(glob.glob(args.data) or [args.data], args.limit)
    X = torch.from_numpy(X)
    S = torch.from_numpy(S)
    R = torch.from_numpy(R)

    n = len(X)
    n_val = max(1, int(n * 0.05))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = Model1(width=args.width, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev}  params={sum(p.numel() for p in model.parameters()):,}"
          f"  train={len(tr_idx)}  val={len(val_idx)}")

    def loss_fn(idx):
        x = X[idx].to(dev)
        s = S[idx].to(dev)
        r = R[idx].to(dev)
        q = model(x).squeeze(1)
        pred = torch.sigmoid(q / SCALE)
        target = lam * torch.sigmoid(s / SCALE) + (1 - lam) * r
        return ((pred - target) ** 2).mean()

    bs = args.batch
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_perm = tr_idx[torch.randperm(len(tr_idx))]
        running = 0.0
        nb = 0
        for i in range(0, len(tr_perm), bs):
            b = tr_perm[i:i + bs]
            opt.zero_grad()
            l = loss_fn(b)
            l.backward()
            opt.step()
            running += l.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = loss_fn(val_idx).item()
        print(f"epoch {ep:2d}  train_loss {running/max(1,nb):.5f}  val_loss {vl:.5f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(),
                "width": args.width, "hidden": args.hidden,
                "n_in": N_FEATURES, "scale": SCALE}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train1.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=2, help="# of 256-wide hidden layers")
    ap.add_argument("--lam", type=float, default=0.7, help="eval/result blend")
    ap.add_argument("--out", default="models/model1.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
