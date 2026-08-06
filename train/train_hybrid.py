"""
Streaming trainer for the hybrid (dense geometry + sparse HalfKA) NNUE.

Each position is encoded twice from a single parsed board: the dense
identity-independent geometry vector and the sparse HalfKA feature lists.  Both
are letter-independent, so the whole model is.  CPU or GPU.

Usage:
    python train/train_hybrid.py --data data/bigdb/bigdb.jsonl \
        --variants data/bigdb/variants.json --king-buckets 32 --dim 256 \
        --epochs 1 --out models/hybrid_bigdb.pt
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
from model_hybrid import HybridNet                      # noqa: E402

SCALE = 361.0


def _collate(samples):
    dense = np.asarray([s[0] for s in samples], dtype=np.float32)
    own_idx, own_off, opp_idx, opp_off, bk, S, R = [], [], [], [], [], [], []
    for _, own, opp, stage, sc, rr in samples:
        own_off.append(len(own_idx)); own_idx.extend(own)
        opp_off.append(len(opp_idx)); opp_idx.extend(opp)
        bk.append(stage); S.append(sc); R.append(rr)
    return (torch.from_numpy(dense),
            torch.tensor(own_idx, dtype=torch.long),
            torch.tensor(own_off, dtype=torch.long),
            torch.tensor(opp_idx, dtype=torch.long),
            torch.tensor(opp_off, dtype=torch.long),
            torch.tensor(bk, dtype=torch.long),
            torch.tensor(S, dtype=torch.float32),
            torch.tensor(R, dtype=torch.float32))


def _emit(buf, variants, reg, king_buckets, batch, rng):
    rng.shuffle(buf)
    samples = []
    last_vm = None
    for line in buf:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("result_stm") is None:
            continue
        vm = variants[rec["vm"]]
        if vm != last_vm:                       # configure cp material per variant
            F.configure_piece_values(vm)
            last_vm = vm
        try:
            board = Board.from_fen(rec["fen"], vm)
            dense = F.encode_board_geo(board, F._betza_map(vm), reg)
            fk = FK.board_features(board, reg, king_buckets, include_kings=True)
        except Exception:
            continue
        samples.append((dense, fk["own"], fk["opp"], fk["stage"],
                        rec["score_cp"], (rec["result_stm"] + 1) / 2.0))
        if len(samples) >= batch:
            yield _collate(samples)
            samples = []
    if samples:
        yield _collate(samples)


def stream_batches(path, variants, reg, king_buckets, batch, chunk_lines, seed):
    rng = random.Random(seed)
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield from _emit(buf, variants, reg, king_buckets, batch, rng)
                buf = []
        if buf:
            yield from _emit(buf, variants, reg, king_buckets, batch, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--king-buckets", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--geo-hidden", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--chunk-lines", type=int, default=500000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="models/hybrid_bigdb.pt")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    F.set_geo_registry(reg)
    dense_in = F.n_features_geo(reg)
    n_feat = FK.num_features(reg, args.king_buckets)
    model = HybridNet(dense_in, n_feat, dim=args.dim, geo_hidden=args.geo_hidden,
                      buckets=3, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev} dense_in={dense_in} king_buckets={args.king_buckets} "
          f"types={reg.num_types} n_features={n_feat:,} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    t0 = time.time(); seen = 0
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0; nb = 0
        for (dn, oi, oo, pi, po, bk, S, R) in stream_batches(
                args.data, variants, reg, args.king_buckets, args.batch,
                args.chunk_lines, seed=ep):
            dn = dn.to(dev); oi, oo = oi.to(dev), oo.to(dev)
            pi, po = pi.to(dev), po.to(dev)
            bk, S, R = bk.to(dev), S.to(dev), R.to(dev)
            opt.zero_grad()
            q = model(dn, oi, oo, pi, po, bk).squeeze(1)
            loss = ((torch.sigmoid(q / SCALE) -
                     (lam * torch.sigmoid(S / SCALE) + (1 - lam) * R)) ** 2).mean()
            loss.backward(); opt.step()
            run += loss.item(); nb += 1; seen += bk.numel()
            if nb % args.log_every == 0:
                el = time.time() - t0
                print(f"ep{ep} batch {nb}  loss {run/nb:.5f}  {seen:,} pos  "
                      f"{seen/el:.0f} pos/s  {el/60:.1f} min", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "dense_in": dense_in,
                    "king_buckets": args.king_buckets, "dim": args.dim,
                    "geo_hidden": args.geo_hidden, "hidden": args.hidden,
                    "n_features": n_feat, "num_types": reg.num_types,
                    "scale": SCALE}, args.out)
        print(f"== epoch {ep} done, mean loss {run/max(1,nb):.5f}, saved {args.out} ==",
              flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min, {seen:,} positions", flush=True)


if __name__ == "__main__":
    main()
