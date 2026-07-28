"""
Compare all four Musketeer NNUE models on one identical train/val split
(first cut of Milestone 6 -- "determine the best NNUE among the 4 models").

This ranks the models by held-out validation loss under matched conditions
(same positions, same split, same epochs/optimiser).  It is a proxy for the
full self-play arena: the true arena (``arena.py``) plays the *deployable*
engine nets against each other, which requires exporting each net to the
Fairy-Stockfish ``.nnue`` format -- pending the architecture/deployability
decision and the GPU-trained final nets.  Validation loss is nonetheless a
sound first screen and is what ``delete_bad_nets.py`` also keys on.

Usage:
    python train/compare.py --data data/processed/train1.jsonl --limit 60000 \
        --epochs 15
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from features import encode_fen, encode_fen_model3     # noqa: E402
from model1 import Model1                              # noqa: E402
from model2 import Model2, fen_gating_finished         # noqa: E402
from model3 import Model3                              # noqa: E402
from model4 import Model4                              # noqa: E402

SCALE = 361.0
VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


def build(patterns, limit):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    X1, X3, S, R, G = [], [], [], [], []
    t0 = time.time()
    done = False
    for fn in files:
        with open(fn, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("result_stm") is None:
                    continue
                X1.append(encode_fen(rec["fen"], VARIANT_MEN))
                X3.append(encode_fen_model3(rec["fen"], VARIANT_MEN))
                S.append(rec["score_cp"]); R.append((rec["result_stm"] + 1) / 2.0)
                G.append(1 if fen_gating_finished(rec["fen"]) else 0)
                if limit and len(X1) >= limit:
                    done = True; break
        if done:
            break
    print(f"encoded {len(X1)} positions in {time.time()-t0:.1f}s")
    return (torch.tensor(np.asarray(X1, np.float32)),
            torch.tensor(np.asarray(X3, np.float32)),
            torch.tensor(np.asarray(S, np.float32)),
            torch.tensor(np.asarray(R, np.float32)),
            torch.tensor(np.asarray(G, np.int64)))


def target(s, r, lam):
    return lam * torch.sigmoid(s / SCALE) + (1 - lam) * r


def run_model(name, model, feat, S, R, G, tr, va, args, gating=False):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bs = args.batch

    def loss(idx):
        if gating:
            g = G[idx]; tot = torch.zeros((), device=dev); cnt = 0
            for sub, post in ((idx[g == 0], False), (idx[g == 1], True)):
                if len(sub) == 0:
                    continue
                x = feat[sub].to(dev)
                q = (model.forward_post(x[:, :64]) if post else model.forward_pre(x)).squeeze(1)
                t = target(S[sub].to(dev), R[sub].to(dev), args.lam)
                tot = tot + ((torch.sigmoid(q / SCALE) - t) ** 2).sum(); cnt += len(sub)
            return tot / max(1, cnt)
        x = feat[idx].to(dev)
        q = model(x).squeeze(1)
        t = target(S[idx].to(dev), R[idx].to(dev), args.lam)
        return ((torch.sigmoid(q / SCALE) - t) ** 2).mean()

    for _ in range(args.epochs):
        model.train()
        p = tr[torch.randperm(len(tr))]
        for i in range(0, len(p), bs):
            opt.zero_grad(); loss(p[i:i + bs]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = loss(va).item()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name:9s} params={params:>9,d}  val_loss={vl:.5f}")
    return name, vl, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/train1.jsonl")
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    args = ap.parse_args()

    X1, X3, S, R, G = build(glob.glob(args.data) or [args.data], args.limit)
    n = len(X1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    nv = max(1, int(n * 0.1))
    va, tr = perm[:nv], perm[nv:]
    print(f"train={len(tr)} val={len(va)}  (identical split for all models)\n")

    results = []
    results.append(run_model("Model1", Model1(width=256, hidden=2), X1, S, R, G, tr, va, args))
    results.append(run_model("Model2", Model2(hidden=1), X1, S, R, G, tr, va, args, gating=True))
    results.append(run_model("Model3", Model3(width=512, hidden=2), X3, S, R, G, tr, va, args))
    results.append(run_model("Model4", Model4(width=256, hidden=4), X1, S, R, G, tr, va, args))

    print("\n=== ranking (lower val_loss = better) ===")
    for rank, (name, vl, params) in enumerate(sorted(results, key=lambda x: x[1]), 1):
        print(f"  {rank}. {name:9s} val_loss={vl:.5f}  params={params:,}")


if __name__ == "__main__":
    main()
