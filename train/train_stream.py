"""
Streaming trainer for a dataset far larger than memory.

Reads the parsed big-database JSONL in shuffled chunks and encodes each batch on
the fly, so we never hold the whole set (or its features) in RAM or on disk. Each
record carries a variant id (`vm`) into variants.json, because every file in the
database uses its own piece rules; we configure the cp piece values per variant
before encoding.

Works on CPU or GPU automatically. On CPU this is slow but bounded; on a GPU it
is far faster and is the recommended way to use a dataset of this size.

Usage:
    python train/train_stream.py --data data/bigdb/bigdb.jsonl \
        --variants data/bigdb/variants.json --encoding mk128 \
        --arch model1 --epochs 2 --out models/model1_bigdb.pt
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
import features as F                                    # noqa: E402
from model1 import Model1                               # noqa: E402
from model2 import Model2                               # noqa: E402
from model3 import Model3                               # noqa: E402
from model4 import Model4                               # noqa: E402
from model512 import Model512                           # noqa: E402

SCALE = 361.0


def encoder_for(name):
    if name == "mk128":
        return F.encode_fen, F.N_FEATURES
    if name == "model3":
        return F.encode_fen_model3, F.N_FEATURES_M3
    if name == "512":
        return F.encode_fen_512, F.N_FEATURES_512
    raise SystemExit("encoding must be mk128 | model3 | 512")


def build_model(arch, n_in):
    if arch == "model1":
        return Model1(n_in=n_in, width=256, hidden=2)
    if arch == "model2":
        return Model2(hidden=1)
    if arch == "model3":
        return Model3(width=512, hidden=2)
    if arch == "model4":
        return Model4(width=256, hidden=4)
    if arch == "model512":
        return Model512(n_in=n_in, width=512, hidden=3)
    raise SystemExit("unknown arch")


def stream_batches(path, variants, encode, batch, chunk_lines, seed):
    """Yield (X, S, R) numpy batches. Chunk-shuffled: read a block, shuffle it,
    encode on the fly. Records reference a per-file variant for encoding."""
    rng = random.Random(seed)
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line)
            if len(buf) >= chunk_lines:
                yield from _emit(buf, variants, encode, batch, rng)
                buf = []
        if buf:
            yield from _emit(buf, variants, encode, batch, rng)


def _emit(buf, variants, encode, batch, rng):
    rng.shuffle(buf)
    X, S, R = [], [], []
    last_vm = None
    for line in buf:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("result_stm") is None:
            continue
        vm = variants[rec["vm"]]
        if vm != last_vm:                         # configure cp values per variant
            F.configure_piece_values(vm)
            last_vm = vm
        try:
            X.append(encode(rec["fen"], vm))
        except Exception:
            continue
        S.append(rec["score_cp"]); R.append((rec["result_stm"] + 1) / 2.0)
        if len(X) >= batch:
            yield (np.asarray(X, np.float32), np.asarray(S, np.float32),
                   np.asarray(R, np.float32))
            X, S, R = [], [], []
    if X:
        yield (np.asarray(X, np.float32), np.asarray(S, np.float32),
               np.asarray(R, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--encoding", default="mk128", help="mk128 | model3 | 512")
    ap.add_argument("--arch", default="model1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--chunk-lines", type=int, default=500000,
                    help="shuffle buffer size (rows read before shuffling)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--out", default="models/model_bigdb.pt")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    variants = json.load(open(args.variants, encoding="utf-8"))
    if args.encoding == "geo":
        # Identity-independent Model 3: index geometry by canonical rule id.
        import betza_id as BID
        reg = BID.registry_from_variants(variants)
        F.set_geo_registry(reg)
        encode, n_in = F.encode_fen_geo, F.n_features_geo(reg)
        print(f"geo encoding: {reg.num_types} canonical piece rules "
              f"-> {n_in} inputs (letter-independent)", flush=True)
    else:
        encode, n_in = encoder_for(args.encoding)
        if args.arch == "model3":
            F.set_model3_types(list("PNBRQKHU"))
    model = build_model(args.arch, n_in).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lam = args.lam
    print(f"device={dev} arch={args.arch} encoding={args.encoding} "
          f"params={sum(p.numel() for p in model.parameters()):,} variants={len(variants)}",
          flush=True)

    t0 = time.time(); seen = 0
    for ep in range(1, args.epochs + 1):
        model.train(); run = 0.0; nb = 0
        for X, S, Rr in stream_batches(args.data, variants, encode, args.batch,
                                       args.chunk_lines, seed=ep):
            x = torch.from_numpy(X).to(dev); s = torch.from_numpy(S).to(dev)
            r = torch.from_numpy(Rr).to(dev)
            opt.zero_grad()
            q = model(x).squeeze(1)
            loss = ((torch.sigmoid(q / SCALE) -
                     (lam * torch.sigmoid(s / SCALE) + (1 - lam) * r)) ** 2).mean()
            loss.backward(); opt.step()
            run += loss.item(); nb += 1; seen += len(X)
            if nb % args.log_every == 0:
                el = time.time() - t0
                print(f"ep{ep} batch {nb}  loss {run/nb:.5f}  "
                      f"{seen:,} pos  {seen/el:.0f} pos/s  {el/60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "arch": args.arch,
                    "encoding": args.encoding, "n_in": n_in, "scale": SCALE},
                   args.out)
        print(f"== epoch {ep} done, mean loss {run/max(1,nb):.5f}, saved {args.out} ==",
              flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min, {seen:,} positions seen", flush=True)


if __name__ == "__main__":
    main()
