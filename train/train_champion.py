"""
Train the champion (best config from the champion experiment) on the COMPLETE
database, with the safeguards a full-scale run needs.

Auto-picks the lowest-val-loss config from models/experiments/champion.json
(hybrid + learnable ReLU + best dim/head) and trains it on every position.

Full-scale additions over the CPU version:
  * runs on GPU automatically (device auto-detected; override with --device);
  * pinned-memory host->device transfer when on CUDA;
  * a cosine learning-rate schedule (--lr-decay-positions) so the run anneals
    instead of hammering a constant rate;
  * periodic validation + early stopping (--val-every, --patience), which is the
    main guard against the overfitting the constant-rate CPU run showed;
  * optional weight decay (--weight-decay);
  * --dense-ft to use a dense feature transformer with a single Adam optimiser
    (simpler and reliable on GPU) instead of the sparse EmbeddingBag + SparseAdam
    used on CPU.

Checkpoints: the latest to <out>, and the best-validation model to
<out>.best.pt. Resumable via --resume-from.
"""
from __future__ import annotations

import argparse
import json
import math
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
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--val-cap", type=int, default=30_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914)
    ap.add_argument("--chunk-lines", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-decay-positions", type=int, default=0,
                    help="cosine-anneal the LR to ~0 over this many positions "
                         "(0 = constant LR)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-every", type=int, default=500,
                    help="validate every N training batches (for early stopping)")
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many validations with no improvement "
                         "(0 = never early-stop)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--dense-ft", action="store_true",
                    help="dense feature transformer + single Adam (recommended on GPU)")
    ap.add_argument("--device", default="", help="cuda / cuda:0 / cpu (default auto)")
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--out", default="models/experiments/champion_FULL.pt")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    keep_awake()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    cfg = pick_champion(args.champion_json)
    print(f"champion: {cfg['_name']} (val {cfg['_val']}) -> {cfg['kind']} "
          f"dim{cfg['dim']} head{cfg['head']} act={cfg['act']}", flush=True)

    sparse = not args.dense_ft
    model = build(cfg, reg, dev, sparse=sparse)
    if args.resume_from and os.path.exists(args.resume_from):
        ck = torch.load(args.resume_from, map_location=dev)
        model.load_state_dict(ck["model"])
        print(f"resumed from {args.resume_from}", flush=True)

    # optimizers: sparse transformer needs SparseAdam (no weight decay there);
    # the dense path uses a single Adam with optional weight decay.
    if sparse:
        ft, other = model.param_groups()
        opt_ft = torch.optim.SparseAdam(ft, lr=args.lr)
        opt_other = torch.optim.Adam(other, lr=args.lr, weight_decay=args.weight_decay)
        opts = [opt_ft, opt_other]
    else:
        opts = [torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)]

    scheds = []
    if args.lr_decay_positions > 0:
        t_max = max(1, args.lr_decay_positions // args.batch)
        scheds = [torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=t_max)
                  for o in opts]

    size = model.nnue_size()
    print(f"device={dev} sparse_ft={sparse} params={size['params']:,} "
          f"~{size['mb']}MB epochs={args.epochs} lr={args.lr} "
          f"wd={args.weight_decay} patience={args.patience}", flush=True)

    val_keep = max(1, args.total_lines // max(1, args.val_cap))
    spec = {"kind": cfg["kind"], "king_buckets": cfg["king_buckets"]}
    pin = (str(dev).startswith("cuda"))

    def make_loader(mode, seed):
        ds = EncodingIterable(args.data, variants, reg, spec, val_keep, 1,
                              chunk_lines=args.chunk_lines, seed=seed, mode=mode)
        return DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                          collate_fn=COLLATE[cfg["kind"]], pin_memory=pin,
                          prefetch_factor=(4 if args.workers else None))

    def validate():
        model.eval(); tot = 0.0; n = 0
        with torch.no_grad():
            for inp, S in make_loader("val", 99):
                inp = _to_dev(inp, dev); S = S.to(dev)
                q = forward(model, inp, cfg["kind"])
                tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
        model.train()
        return tot / max(1, n)

    def save(path, extra):
        torch.save({"model": model.state_dict(), "cfg": cfg, "scale": SCALE,
                    **extra}, path)

    best = None; bad = 0; stop = False
    t0 = time.time(); seen = 0
    for ep in range(1, args.epochs + 1):
        if stop:
            break
        model.train(); run = 0.0; nb = 0
        for inp, S in make_loader("train", ep):
            inp = _to_dev(inp, dev); S = S.to(dev)
            for o in opts:
                o.zero_grad()
            q = forward(model, inp, cfg["kind"])
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward()
            for o in opts:
                o.step()
            for s in scheds:
                s.step()
            run += loss.item(); nb += 1; seen += S.numel()
            if nb % args.log_every == 0:
                el = time.time() - t0
                lo, hi = model.activation_bounds()
                lr_now = opts[0].param_groups[0]["lr"]
                print(f"  ep{ep}: {seen:,} pos  {seen/el:.0f} pos/s  train {run/nb:.5f}  "
                      f"lr {lr_now:.2e}  clip[{lo:.2f},{hi:.2f}]  {el/60:.1f} min",
                      flush=True)
            if nb % args.save_every == 0:
                save(args.out, {"epoch": ep - 1, "partial_pos": seen, "full_data": True})
            if args.val_every and nb % args.val_every == 0:
                v = validate()
                if best is None or v < best - 1e-6:
                    best = v; bad = 0
                    save(args.out + ".best.pt",
                         {"epoch": ep, "val_loss": v, "full_data": True, "best": True})
                    print(f"    val {v:.5f}  (new best, saved best.pt)", flush=True)
                else:
                    bad += 1
                    print(f"    val {v:.5f}  (no improve {bad}/{args.patience})", flush=True)
                    if args.patience and bad >= args.patience:
                        print("    early stopping", flush=True); stop = True; break
            if args.max_positions and seen >= args.max_positions:
                stop = True; break
        v = validate()
        if best is None or v < best - 1e-6:
            best = v; save(args.out + ".best.pt",
                           {"epoch": ep, "val_loss": v, "full_data": True, "best": True})
        save(args.out, {"epoch": ep, "val_loss": v, "full_data": True})
        print(f"== epoch {ep}/{args.epochs}: {seen:,} pos, VAL {v:.5f} "
              f"(best {best:.5f}), saved {args.out} ==", flush=True)
    print(f"CHAMPION TRAINING DONE, best VAL {best:.5f} -> {args.out}.best.pt", flush=True)


if __name__ == "__main__":
    main()
