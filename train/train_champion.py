"""
Train the champion (best config from the champion experiment) on the COMPLETE
134M-position database.

Auto-picks the lowest-val-loss config from models/experiments/champion.json
(hybrid + learnable ReLU + best dim/head), then trains it on every position with
SparseAdam on the feature transformer and Adam on the rest. Checkpoints every
--save-every batches and at each epoch end, so a partial run always leaves a
usable model. Resumable via --resume-from.
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
from experiments import build, forward, _to_dev, keep_awake   # noqa: E402


def pick_champion(path):
    rows = [x for x in json.load(open(path)) if "val_loss" in x]
    if not rows:
        raise SystemExit("no completed champion configs to pick from")
    best = min(rows, key=lambda x: x["val_loss"])
    return {"kind": best["kind"], "dim": best["dim"], "head": tuple(best["head"]),
            "act": best["act"], "king_buckets": 16, "_val": best["val_loss"],
            "_name": best["name"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion-json", default="models/experiments/champion.json")
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--val-cap", type=int, default=30_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914)
    ap.add_argument("--chunk-lines", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--out", default="models/experiments/champion_FULL.pt")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    keep_awake()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    cfg = pick_champion(args.champion_json)
    print(f"champion: {cfg['_name']} (val {cfg['_val']}) -> {cfg['kind']} "
          f"dim{cfg['dim']} head{cfg['head']} act={cfg['act']}", flush=True)

    model = build(cfg, reg, dev)
    if args.resume_from and os.path.exists(args.resume_from):
        ck = torch.load(args.resume_from, map_location=dev)
        model.load_state_dict(ck["model"])
        print(f"resumed from {args.resume_from}", flush=True)
    ft, other = model.param_groups()
    opt_ft = torch.optim.SparseAdam(ft, lr=args.lr)
    opt_other = torch.optim.Adam(other, lr=args.lr)
    size = model.nnue_size()
    print(f"device={dev} params={size['params']:,} ~{size['mb']}MB epochs={args.epochs}",
          flush=True)

    val_keep = max(1, args.total_lines // max(1, args.val_cap))
    spec = {"kind": cfg["kind"], "king_buckets": cfg["king_buckets"]}

    def make_loader(mode, seed):
        ds = EncodingIterable(args.data, variants, reg, spec, val_keep, 1,
                              chunk_lines=args.chunk_lines, seed=seed, mode=mode)
        return DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                          collate_fn=COLLATE[cfg["kind"]],
                          prefetch_factor=(4 if args.workers else None))

    def save(tag, extra):
        torch.save({"model": model.state_dict(), "cfg": cfg, "scale": SCALE,
                    **extra}, args.out)

    t0 = time.time(); best = None
    for ep in range(1, args.epochs + 1):
        model.train(); seen = 0; run = 0.0; nb = 0
        for inp, S in make_loader("train", ep):
            inp = _to_dev(inp, dev); S = S.to(dev)
            opt_ft.zero_grad(); opt_other.zero_grad()
            q = forward(model, inp, cfg["kind"])
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward(); opt_ft.step(); opt_other.step()
            run += loss.item(); nb += 1; seen += S.numel()
            if nb % args.log_every == 0:
                el = time.time() - t0
                lo, hi = model.activation_bounds()
                print(f"  ep{ep}: {seen:,} pos  {seen/el:.0f} pos/s  train {run/nb:.5f}  "
                      f"clip[{lo:.2f},{hi:.2f}]  {el/60:.1f} min", flush=True)
            if nb % args.save_every == 0:
                save("partial", {"epoch": ep - 1, "partial_pos": seen, "full_data": True})
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for inp, S in make_loader("val", 99):
                inp = _to_dev(inp, dev); S = S.to(dev)
                q = forward(model, inp, cfg["kind"])
                tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
        vloss = tot / max(1, n)
        best = vloss if best is None else min(best, vloss)
        lo, hi = model.activation_bounds()
        save("epoch", {"epoch": ep, "val_loss": vloss, "clip": [lo, hi], "full_data": True})
        print(f"== epoch {ep}/{args.epochs}: {seen:,} pos, VAL {vloss:.5f} "
              f"(best {best:.5f}), clip[{lo:.2f},{hi:.2f}], saved {args.out} ==", flush=True)
    print(f"CHAMPION FULL-DATA DONE, best VAL {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
