"""
Full-data training driver: trains all four models on ALL 835k positions and
writes real checkpoints to models/. Run in the background; progress is logged.

    python train/run_full.py --epochs 20 --batch 8192
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

DATA = "data/processed/uni_hawk_training_nnue-*.jsonl"
MODELS = ["model1", "model2", "model3", "model4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help="which models to train (e.g. model2 model3 model4)")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads per model — keep low so the laptop stays cool")
    ap.add_argument("--throttle-ms", type=int, default=0,
                    help="sleep this many ms between batches (extra gentle)")
    args = ap.parse_args()

    # Gentle-mode: cap the CPU threads the math libraries use so we don't peg
    # every core (that's what makes the fans roar).  Inherited by subprocesses.
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = str(args.threads)
    env["MK_TRAIN_THREADS"] = str(args.threads)
    env["MK_THROTTLE_MS"] = str(args.throttle_ms)
    print(f"gentle mode: {args.threads} threads/model, "
          f"throttle {args.throttle_ms}ms/batch\n", flush=True)

    t0 = time.time()
    for m in args.models:
        print(f"\n{'='*60}\n=== FULL TRAIN {m}  (epochs={args.epochs})\n{'='*60}",
              flush=True)
        ts = time.time()
        r = subprocess.run(
            [sys.executable, f"train/{m}.py", "--data", DATA,
             "--epochs", str(args.epochs), "--batch", str(args.batch),
             "--lr", str(args.lr), "--out", f"models/{m}.pt"],
            capture_output=True, text=True, env=env)
        print(r.stdout, flush=True)
        if r.returncode != 0:
            print(f"!! {m} FAILED:\n{r.stderr}", flush=True)
        else:
            print(f">> {m} done in {(time.time()-ts)/60:.1f} min", flush=True)
    print(f"\nALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
