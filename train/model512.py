"""
Model 512 -- the client's proposed NNUE architecture.

    input(512)  ->  512 -> 512 -> 512 -> 1   (clipped ReLU)

Input layout (features.encode_fen_512): 64 material (cp values) + 16 gating +
432 Betza geometry (atoms, B/R/Q sliders, controlled squares D1..D4, colour
bound, directions), for 24 piece types weighted by signed on-board count.

The input width is configurable through --keep, so the 256 / 384 / 512
ablations the client asked about are a one-line change.

Usage:
    python train/model512.py --data data/processed/reeval_ii_jj.jsonl \
        --variant-men "N:N;B:B;...;I:ZD;J:ZW;K:KisO2" --epochs 25 --out models/model512_asym.pt
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
from features import (encode_fen_512, N_FEATURES_512,            # noqa: E402
                      configure_piece_values)

SCALE = 361.0
SYM_VM = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
          "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


class Model512(nn.Module):
    def __init__(self, n_in=N_FEATURES_512, width=512, hidden=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(n_in, width)] + [nn.Linear(width, width) for _ in range(hidden)])
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        for lin in self.layers:
            x = torch.clamp(lin(x), 0.0, 1.0)
        return self.output(x)


# directions feature index within each 24 x 18 geometry block (last of 18)
DIR_IDX = [80 + ti * 18 + 17 for ti in range(24)]


def load_dataset(patterns, limit, variant_men, keep, no_dir=False):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no files matched {patterns}")
    key = os.path.join(os.path.dirname(files[0]),
                       f".cache512_{abs(hash((tuple(files), limit, variant_men, keep, no_dir))) & 0xffffffff}.npz")
    if os.path.exists(key):
        d = np.load(key); print(f"loaded cache {key}: {len(d['X'])} rows")
        return d["X"], d["S"], d["R"]
    configure_piece_values(variant_men)
    X, S, R = [], [], []
    t0 = time.time()
    for fn in files:
        for line in open(fn, encoding="utf-8"):
            rec = json.loads(line)
            if rec.get("result_stm") is None:
                continue
            v = encode_fen_512(rec["fen"], variant_men)
            if no_dir:
                for di in DIR_IDX:
                    v[di] = 0.0
            X.append(v[:keep])
            S.append(rec["score_cp"]); R.append((rec["result_stm"] + 1) / 2.0)
            if limit and len(X) >= limit:
                break
        if limit and len(X) >= limit:
            break
    X = np.asarray(X, np.float32); S = np.asarray(S, np.float32); R = np.asarray(R, np.float32)
    print(f"encoded {len(X)} positions ({keep} inputs) in {time.time()-t0:.1f}s")
    np.savez_compressed(key, X=X, S=S, R=R)
    return X, S, R


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keep = args.keep or N_FEATURES_512
    X, S, R = load_dataset(glob.glob(args.data) or [args.data], args.limit,
                           args.variant_men or SYM_VM, keep, args.no_dir)
    X = torch.from_numpy(X); S = torch.from_numpy(S); R = torch.from_numpy(R)
    n = len(X); perm = torch.randperm(n); nv = max(1, int(n * 0.05))
    va, tr = perm[:nv], perm[nv:]
    model = Model512(n_in=keep, width=args.width, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr); lam = args.lam
    print(f"device={dev} inputs={keep} params={sum(p.numel() for p in model.parameters()):,} "
          f"train={len(tr)} val={len(va)}")

    def loss_fn(idx):
        x = X[idx].to(dev); s = S[idx].to(dev); r = R[idx].to(dev)
        q = model(x).squeeze(1)
        return ((torch.sigmoid(q / SCALE) -
                 (lam * torch.sigmoid(s / SCALE) + (1 - lam) * r)) ** 2).mean()

    bs = args.batch
    for ep in range(1, args.epochs + 1):
        model.train(); p = tr[torch.randperm(len(tr))]; run = 0.0; nb = 0
        for i in range(0, len(p), bs):
            opt.zero_grad(); l = loss_fn(p[i:i + bs]); l.backward(); opt.step()
            run += l.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vl = loss_fn(va).item()
        print(f"epoch {ep:2d}  train_loss {run/max(1,nb):.5f}  val_loss {vl:.5f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "width": args.width, "hidden": args.hidden,
                "n_in": keep, "scale": SCALE, "arch": "model512"}, args.out)
    print("saved", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/reeval_ii_jj.jsonl")
    ap.add_argument("--variant-men", dest="variant_men", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--keep", type=int, default=0,
                    help="use only the first N inputs (512 default, 384/256 for ablations)")
    ap.add_argument("--no-directions", dest="no_dir", action="store_true")
    ap.add_argument("--out", default="models/model512.pt")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
