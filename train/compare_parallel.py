"""
Nine-way architecture comparison, encoding in parallel across all CPU cores.

Same idea as compare_models.py (equal budget, held-out validation, ranked table,
resumable) but the feature encoding runs in DataLoader worker processes so the
machine's cores are all busy — ~3x faster on this 4-core box.  Every model is
letter-independent (canonical Betza id throughout).

Usage:
    python train/compare_parallel.py --train-positions 10000000 --workers 4
    python train/compare_parallel.py --only geo,halfka,hybrid
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
import features as F                                    # noqa: E402
import features_halfka as FK                            # noqa: E402
from parallel import EncodingIterable, COLLATE, SCALE   # noqa: E402
from model1 import Model1                               # noqa: E402
from model2 import Model2                               # noqa: E402
from model3 import Model3                               # noqa: E402
from model4 import Model4                               # noqa: E402
from model512 import Model512                           # noqa: E402
from model_halfka import HalfKANet                      # noqa: E402
from model_hybrid import HybridNet                      # noqa: E402

SPECS = [
    {"kind": "dense", "name": "model1_mk128", "encoding": "mk128", "arch": "model1"},
    {"kind": "dense", "name": "model2_mk128", "encoding": "mk128", "arch": "model2"},
    {"kind": "dense", "name": "model3_legacy", "encoding": "model3", "arch": "model3"},
    {"kind": "dense", "name": "model4_mk128", "encoding": "mk128", "arch": "model4"},
    {"kind": "dense", "name": "geo", "encoding": "geo", "arch": "model1"},
    {"kind": "dense", "name": "net512", "encoding": "512", "arch": "model512"},
    {"kind": "halfkp", "name": "halfkp", "king_buckets": 16},
    {"kind": "halfka", "name": "halfka", "king_buckets": 16},
    {"kind": "hybrid", "name": "hybrid", "king_buckets": 16},
]


def _keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("keep-awake enabled", flush=True)
    except Exception:
        pass


def build_model(spec, reg, dim, dev):
    kind = spec["kind"]
    if kind == "dense":
        enc, arch = spec["encoding"], spec["arch"]
        if enc == "mk128":
            n = F.N_FEATURES
        elif enc == "512":
            n = F.N_FEATURES_512
        elif enc == "model3":
            F.set_model3_types(list("PNBRQKHU")); n = F.N_FEATURES_M3
        elif enc == "geo":
            F.set_geo_registry(reg); n = F.n_features_geo(reg)
        m = {"model1": lambda: Model1(n_in=n, width=256, hidden=2),
             "model2": lambda: Model2(hidden=1),
             "model3": lambda: Model3(width=512, hidden=2),
             "model4": lambda: Model4(width=256, hidden=4),
             "model512": lambda: Model512(n_in=n, width=512, hidden=3)}[arch]()
        return m.to(dev)
    if kind in ("halfka", "halfkp"):
        nf = FK.num_features(reg, spec["king_buckets"])
        return HalfKANet(nf, dim=dim, buckets=3, hidden=32).to(dev)
    F.set_geo_registry(reg)                              # hybrid
    di = F.n_features_geo(reg)
    nf = FK.num_features(reg, spec["king_buckets"])
    return HybridNet(di, nf, dim=dim, geo_hidden=64, buckets=3, hidden=32).to(dev)


def forward(spec, model, inp):
    if spec["kind"] == "dense":
        if spec["arch"] == "model2":
            x = inp
            post = x[:, 64:128].abs().sum(1) == 0
            q = x.new_zeros(x.size(0), 1)
            if (~post).any():
                q[~post] = model.forward_pre(x[~post])
            if post.any():
                q[post] = model.forward_post(x[post][:, :64])
            return q.squeeze(1)
        return model(inp).squeeze(1)
    return model(*inp).squeeze(1)


def _to_dev(inp, dev):
    if isinstance(inp, (tuple, list)):
        return tuple(t.to(dev) for t in inp)
    return inp.to(dev)


def loader(spec, data, variants, reg, val_keep, train_keep, mode, batch,
           workers, chunk_lines, seed):
    ds = EncodingIterable(data, variants, reg, spec, val_keep, train_keep,
                          chunk_lines=chunk_lines, seed=seed, mode=mode)
    return DataLoader(ds, batch_size=batch, num_workers=workers,
                      collate_fn=COLLATE[spec["kind"]],
                      persistent_workers=False,
                      prefetch_factor=(4 if workers else None))


def run_one(spec, args, variants, reg, val_keep, train_keep, dev):
    model = build_model(spec, reg, args.dim, dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {spec['name']} | params {params:,} | budget {args.train_positions:,} ===",
          flush=True)
    t0 = time.time(); seen = 0; done = False
    for ep in range(1, args.epochs + 1):
        if done:
            break
        model.train()
        dl = loader(spec, args.data, variants, reg, val_keep, train_keep,
                    "train", args.batch, args.workers, args.chunk_lines, seed=ep)
        for inp, S in dl:
            inp = _to_dev(inp, dev); S = S.to(dev)
            opt.zero_grad()
            q = forward(spec, model, inp)
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward(); opt.step()
            seen += S.numel()
            if seen // args.batch % args.log_every == 0:
                el = time.time() - t0
                print(f"  {spec['name']}: {seen:,} pos  {seen/el:.0f} pos/s  "
                      f"train {loss.item():.5f}  {el/60:.1f} min", flush=True)
            if seen >= args.train_positions:
                done = True
                break
        del dl
    # validation
    model.eval(); tot = 0.0; n = 0
    vdl = loader(spec, args.data, variants, reg, val_keep, train_keep,
                 "val", args.batch, args.workers, args.chunk_lines, seed=99)
    with torch.no_grad():
        for inp, S in vdl:
            inp = _to_dev(inp, dev); S = S.to(dev)
            q = forward(spec, model, inp)
            tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
    vloss = tot / max(1, n)
    dt = (time.time() - t0) / 60
    print(f"--- {spec['name']}: VAL {vloss:.5f}  ({seen:,} pos, {dt:.1f} min, "
          f"{params:,} params, val_n={n:,}) ---", flush=True)
    torch.save({"model": model.state_dict(), "spec": spec, "scale": SCALE},
               os.path.join(args.model_dir, f"{spec['name']}.pt"))
    return {"name": spec["name"], "val_loss": round(vloss, 5), "params": params,
            "train_pos": seen, "minutes": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--train-positions", type=int, default=10_000_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--val-cap", type=int, default=50_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914)
    ap.add_argument("--chunk-lines", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="models/compare_results.json")
    ap.add_argument("--model-dir", default="models/bigdb")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    _keep_awake()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.model_dir, exist_ok=True)
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    val_keep = max(1, args.total_lines // max(1, args.val_cap))
    need = max(1, args.train_positions * max(1, args.epochs))
    train_keep = max(1, args.total_lines // need)
    print(f"device={dev} workers={args.workers} types={reg.num_types} "
          f"budget={args.train_positions:,} val_stride={val_keep} "
          f"train_stride={train_keep}", flush=True)

    specs = SPECS
    if args.only:
        want = set(s.strip() for s in args.only.split(","))
        specs = [s for s in SPECS if s["name"] in want]

    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    done_names = {r["name"] for r in results}
    for spec in specs:
        if spec["name"] in done_names:
            print(f"skip {spec['name']} (already done)", flush=True)
            continue
        try:
            res = run_one(spec, args, variants, reg, val_keep, train_keep, dev)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"name": spec["name"], "error": f"{type(e).__name__}: {e}"}
        results.append(res)
        json.dump(results, open(args.out, "w"), indent=2)

    ranked = sorted((r for r in results if "val_loss" in r), key=lambda r: r["val_loss"])
    print("\n" + "=" * 66)
    print(f"{'model':<16}{'val_loss':>10}{'params':>14}{'min':>8}")
    print("-" * 66)
    for r in ranked:
        print(f"{r['name']:<16}{r['val_loss']:>10.5f}{r['params']:>14,}{r['minutes']:>8.1f}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<16}   ERROR: {r['error']}")
    print("=" * 66)
    print(f"results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
