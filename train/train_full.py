"""
Train ONE model on the COMPLETE big-database (all 134M positions), several epochs.

Meant to run after compare_parallel.py: by default it reads the comparison
results, picks the winning architecture (lowest validation loss), and trains that
one on every position in the dataset, encoding in parallel across all CPU cores.
A small held-out set (same stride split as the comparison) is used only to report
validation loss per epoch; a checkpoint is saved after every epoch, so progress
is never lost.

Usage:
    python train/train_full.py                       # auto-pick the winner
    python train/train_full.py --spec-name geo --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import betza_id as BID                                  # noqa: E402
from parallel import EncodingIterable, COLLATE, SCALE   # noqa: E402
from compare_parallel import (SPECS, build_model, forward, _to_dev,  # noqa: E402
                              _keep_awake)


def pick_winner(results_path):
    d = json.load(open(results_path))
    ok = [r for r in d if "val_loss" in r]
    if not ok:
        raise SystemExit("no completed models in results to pick a winner from")
    best = min(ok, key=lambda r: r["val_loss"])
    return best["name"], best["val_loss"]


def spec_by_name(name):
    for s in SPECS:
        if s["name"] == name:
            return s
    raise SystemExit(f"unknown spec name: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--spec-name", default="auto",
                    help="'auto' picks the winner from --results, or a model name")
    ap.add_argument("--results", default="models/compare_results.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--val-cap", type=int, default=50_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914)
    ap.add_argument("--chunk-lines", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--out-dir", default="models/bigdb")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200,
                    help="also checkpoint every N batches (mid-epoch safety)")
    ap.add_argument("--resume-from", default="",
                    help="load model weights from this checkpoint before training")
    args = ap.parse_args()

    _keep_awake()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)

    if args.spec_name == "auto":
        name, vloss = pick_winner(args.results)
        print(f"winner from {args.results}: {name} (val {vloss})", flush=True)
    else:
        name = args.spec_name
    spec = spec_by_name(name)

    val_keep = max(1, args.total_lines // max(1, args.val_cap))
    train_keep = 1                                    # ALL non-val lines = full data
    model = build_model(spec, reg, args.dim, dev)
    if args.resume_from and os.path.exists(args.resume_from):
        ck = torch.load(args.resume_from, map_location=dev)
        model.load_state_dict(ck["model"])
        print(f"resumed weights from {args.resume_from} "
              f"(prev partial_pos={ck.get('partial_pos')}, val={ck.get('val_loss')})",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    out = os.path.join(args.out_dir, f"{name}_FULL.pt")
    print(f"device={dev} workers={args.workers} FULL-DATA training '{name}' "
          f"params={params:,} epochs={args.epochs} val_stride={val_keep}\n"
          f"checkpoint -> {out}", flush=True)

    def make_loader(mode, seed):
        ds = EncodingIterable(args.data, variants, reg, spec, val_keep, train_keep,
                              chunk_lines=args.chunk_lines, seed=seed, mode=mode)
        return DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                          collate_fn=COLLATE[spec["kind"]],
                          persistent_workers=False,
                          prefetch_factor=(4 if args.workers else None))

    t0 = time.time(); best = None
    for ep in range(1, args.epochs + 1):
        model.train(); seen = 0; run = 0.0; nb = 0
        for inp, S in make_loader("train", seed=ep):
            inp = _to_dev(inp, dev); S = S.to(dev)
            opt.zero_grad()
            q = forward(spec, model, inp)
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward(); opt.step()
            run += loss.item(); nb += 1; seen += S.numel()
            if nb % args.log_every == 0:
                el = time.time() - t0
                print(f"  ep{ep}: {seen:,} pos  {seen/el:.0f} pos/s  "
                      f"train {run/nb:.5f}  {el/60:.1f} min", flush=True)
            if nb % args.save_every == 0:               # mid-epoch safety checkpoint
                torch.save({"model": model.state_dict(), "spec": spec,
                            "scale": SCALE, "epoch": ep - 1, "partial_pos": seen,
                            "full_data": True}, out)
        # validation
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for inp, S in make_loader("val", seed=99):
                inp = _to_dev(inp, dev); S = S.to(dev)
                q = forward(spec, model, inp)
                tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
        vloss = tot / max(1, n)
        best = vloss if best is None else min(best, vloss)
        torch.save({"model": model.state_dict(), "spec": spec, "scale": SCALE,
                    "epoch": ep, "val_loss": vloss, "full_data": True}, out)
        print(f"== epoch {ep}/{args.epochs} done: {seen:,} pos, VAL {vloss:.5f} "
              f"(best {best:.5f}), saved {out} ==", flush=True)
    print(f"FULL-DATA TRAINING DONE for '{name}' in {(time.time()-t0)/60:.1f} min, "
          f"best VAL {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
