"""
Streaming trainer for the HalfKP / HalfKA (canonical-id) NNUE.

Reads the parsed big-database JSONL in shuffled chunks, turns each position into
its sparse active-feature lists on the fly (own + opponent perspective) and
trains the bucketed HalfKA network.  No dense feature matrix is ever built, so
the memory footprint stays tiny regardless of dataset size.  Works on CPU or GPU.

  --mode halfka   emit every piece (king included as a piece)   [default]
  --mode halfkp   king anchors only (classic HalfKP)
  --king-buckets  coarsen the king square (<=64); 64 == exact HalfK

Usage:
    python train/train_halfka.py --data data/bigdb/bigdb.jsonl \
        --variants data/bigdb/variants.json --mode halfka \
        --king-buckets 32 --dim 256 --epochs 1 --out models/halfka_bigdb.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import betza_id as BID                                  # noqa: E402
import features_halfka as FK                            # noqa: E402
from musketeer import Board                             # noqa: E402
from model_halfka import HalfKANet                      # noqa: E402

SCALE = 361.0


def _emit(buf, variants, reg, include_kings, king_buckets, batch, rng):
    rng.shuffle(buf)
    samples = []
    for line in buf:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("result_stm") is None:
            continue
        try:
            board = Board.from_fen(rec["fen"], variants[rec["vm"]])
            f = FK.board_features(board, reg, king_buckets, include_kings)
        except Exception:
            continue
        if not f["own"] and not f["opp"]:
            continue
        samples.append((f["own"], f["opp"], f["stage"],
                        rec["score_cp"], (rec["result_stm"] + 1) / 2.0))
        if len(samples) >= batch:
            yield _collate(samples)
            samples = []
    if samples:
        yield _collate(samples)


def _collate(samples):
    own_idx, own_off, opp_idx, opp_off = [], [], [], []
    bk, S, R = [], [], []
    for own, opp, stage, s, r in samples:
        own_off.append(len(own_idx)); own_idx.extend(own)
        opp_off.append(len(opp_idx)); opp_idx.extend(opp)
        bk.append(stage); S.append(s); R.append(r)
    return (torch.tensor(own_idx, dtype=torch.long),
            torch.tensor(own_off, dtype=torch.long),
            torch.tensor(opp_idx, dtype=torch.long),
            torch.tensor(opp_off, dtype=torch.long),
            torch.tensor(bk, dtype=torch.long),
            torch.tensor(S, dtype=torch.float32),
            torch.tensor(R, dtype=torch.float32))


def stream_batches(path, variants, reg, include_kings, king_buckets, batch,
                   chunk_lines, seed):
    rng = random.Random(seed)
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield from _emit(buf, variants, reg, include_kings,
                                 king_buckets, batch, rng)
                buf = []
        if buf:
            yield from _emit(buf, variants, reg, include_kings,
                             king_buckets, batch, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--mode", default="halfka", choices=["halfka", "halfkp"])
    ap.add_argument("--king-buckets", type=int, default=32)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--chunk-lines", type=int, default=500000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="models/halfka_bigdb.pt")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    include_kings = (args.mode == "halfka")
    n_feat = FK.num_features(reg, args.king_buckets)
    model = HalfKANet(n_feat, dim=args.dim, buckets=3, hidden=args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev} mode={args.mode} king_buckets={args.king_buckets} "
          f"types={reg.num_types} n_features={n_feat:,} "
          f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    t0 = time.time(); seen = 0
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0; nb = 0
        for (oi, oo, pi, po, bk, S, R) in stream_batches(
                args.data, variants, reg, include_kings, args.king_buckets,
                args.batch, args.chunk_lines, seed=ep):
            oi, oo = oi.to(dev), oo.to(dev)
            pi, po = pi.to(dev), po.to(dev)
            bk, S, R = bk.to(dev), S.to(dev), R.to(dev)
            opt.zero_grad()
            q = model(oi, oo, pi, po, bk).squeeze(1)
            loss = ((torch.sigmoid(q / SCALE) -
                     (lam * torch.sigmoid(S / SCALE) + (1 - lam) * R)) ** 2).mean()
            loss.backward(); opt.step()
            run += loss.item(); nb += 1; seen += bk.numel()
            if nb % args.log_every == 0:
                el = time.time() - t0
                print(f"ep{ep} batch {nb}  loss {run/nb:.5f}  {seen:,} pos  "
                      f"{seen/el:.0f} pos/s  {el/60:.1f} min", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"model": model.state_dict(), "mode": args.mode,
                    "king_buckets": args.king_buckets, "dim": args.dim,
                    "hidden": args.hidden, "n_features": n_feat,
                    "num_types": reg.num_types, "scale": SCALE}, args.out)
        print(f"== epoch {ep} done, mean loss {run/max(1,nb):.5f}, saved {args.out} ==",
              flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min, {seen:,} positions", flush=True)


if __name__ == "__main__":
    main()
