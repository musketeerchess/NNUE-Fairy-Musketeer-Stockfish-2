"""
Zied's experiment program, run on the big database with parallel encoding.

Named experiments (--exp):
  arch     the 10 Stockfish-style architectures (dim x2 -> head -> 1), on HalfKA
  relu     clipped ReLU [0,1] vs learnable asymmetric clip, on a fixed arch
  models   HalfKA vs hybrid vs Ultra (gating+Betza) at a fixed arch
  scaling  same model on 800k vs 1.6M positions (dataset-duplication test)
  crossarmy  one model trained/validated separately on the top-N armies

Every run reports validation loss, parameter count, estimated .nnue size, and,
for the learnable activation, the clip interval it settled on. Results are saved
incrementally and can be resumed. The feature transformer trains with SparseAdam
(only touched rows update) and everything else with Adam, so even the dim-1024
transformers are tractable on CPU.
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
from model_nnue import NNUENet                          # noqa: E402


def keep_awake():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def build(cfg, reg, dev):
    n_feat = FK.num_features(reg, cfg["king_buckets"])
    if cfg["kind"] in ("halfka", "halfkp"):
        dense_in = 0
    elif cfg["kind"] == "hybrid":
        F.set_geo_registry(reg); dense_in = F.n_features_geo(reg)
    else:  # ultra
        F.set_geo_registry(reg); dense_in = F.n_features_gatebetza(reg)
    return NNUENet(n_feat, dim=cfg["dim"], head=tuple(cfg["head"]), buckets=3,
                   dense_in=dense_in, act=cfg["act"]).to(dev)


def forward(model, inp, kind):
    if kind in ("hybrid", "ultra"):
        dense, oi, oo, pi, po, bk = inp
        return model(oi, oo, pi, po, bk, dense=dense).squeeze(1)
    oi, oo, pi, po, bk = inp
    return model(oi, oo, pi, po, bk).squeeze(1)


def _to_dev(inp, dev):
    return tuple(t.to(dev) for t in inp)


def loader(cfg, args, reg, variants, val_keep, train_keep, mode, seed, vm_filter):
    spec = {"kind": cfg["kind"], "king_buckets": cfg["king_buckets"]}
    ds = EncodingIterable(args.data, variants, reg, spec, val_keep, train_keep,
                          chunk_lines=args.chunk_lines, seed=seed, mode=mode,
                          vm_filter=vm_filter)
    return DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                      collate_fn=COLLATE[cfg["kind"]],
                      prefetch_factor=(4 if args.workers else None))


def run_cfg(cfg, args, reg, variants, val_keep, train_keep, dev, vm_filter=None):
    model = build(cfg, reg, dev)
    size = model.nnue_size()
    ft, other = model.param_groups()
    opt_ft = torch.optim.SparseAdam(ft, lr=args.lr)
    opt_other = torch.optim.Adam(other, lr=args.lr)
    budget = cfg.get("budget", args.budget)
    print(f"\n=== {cfg['name']} | {cfg['kind']} dim{cfg['dim']} head{tuple(cfg['head'])} "
          f"act={cfg['act']} | params {size['params']:,} ~{size['mb']}MB | budget {budget:,} ===",
          flush=True)
    t0 = time.time(); seen = 0; done = False
    for ep in range(1, args.epochs + 1):
        if done:
            break
        model.train()
        dl = loader(cfg, args, reg, variants, val_keep, train_keep, "train", ep, vm_filter)
        for inp, S in dl:
            inp = _to_dev(inp, dev); S = S.to(dev)
            opt_ft.zero_grad(); opt_other.zero_grad()
            q = forward(model, inp, cfg["kind"])
            loss = ((torch.sigmoid(q / SCALE) - S) ** 2).mean()
            loss.backward(); opt_ft.step(); opt_other.step()
            seen += S.numel()
            if seen // args.batch % args.log_every == 0:
                el = time.time() - t0
                print(f"  {cfg['name']}: {seen:,} pos  {seen/el:.0f} pos/s  "
                      f"train {loss.item():.5f}  {el/60:.1f} min", flush=True)
            if seen >= budget:
                done = True
                break
        del dl
    model.eval(); tot = 0.0; n = 0
    vdl = loader(cfg, args, reg, variants, val_keep, train_keep, "val", 99, vm_filter)
    with torch.no_grad():
        for inp, S in vdl:
            inp = _to_dev(inp, dev); S = S.to(dev)
            q = forward(model, inp, cfg["kind"])
            tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
    vloss = tot / max(1, n)
    lo, hi = model.activation_bounds()
    dt = (time.time() - t0) / 60
    res = {"name": cfg["name"], "kind": cfg["kind"], "dim": cfg["dim"],
           "head": list(cfg["head"]), "act": cfg["act"],
           "val_loss": round(vloss, 5), "params": size["params"],
           "nnue_mb": size["mb"], "clip_lo": round(lo, 3), "clip_hi": round(hi, 3),
           "train_pos": seen, "val_n": n, "minutes": round(dt, 1)}
    print(f"--- {cfg['name']}: VAL {vloss:.5f} | {size['mb']}MB | "
          f"clip[{lo:.2f},{hi:.2f}] | {dt:.1f} min ---", flush=True)
    return res


# --------------------------------------------------------------------------- #
# Experiment definitions
# --------------------------------------------------------------------------- #
ARCH_10 = [
    ("A1_256_64_32", 256, (64, 32)), ("A2_512_64_32", 512, (64, 32)),
    ("A3_1024_64_32", 1024, (64, 32)), ("A4_256_64_64", 256, (64, 64)),
    ("A5_512_64_64", 512, (64, 64)), ("A6_1024_64_64", 1024, (64, 64)),
    ("A7_1024_8_32", 1024, (8, 32)), ("A8_1024_16_32", 1024, (16, 32)),
    ("A9_1024_32_32", 1024, (32, 32)), ("A10_1024_8_64", 1024, (8, 64)),
]


def exp_arch(args):
    return [{"name": n, "kind": "halfka", "dim": d, "head": h,
             "act": "clip", "king_buckets": args.king_buckets} for n, d, h in ARCH_10]


def exp_relu(args):
    base = dict(kind="halfka", dim=256, head=(64, 32), king_buckets=args.king_buckets)
    return [{**base, "name": "relu_clip[0,1]", "act": "clip"},
            {**base, "name": "relu_learnable", "act": "learn"}]


def exp_models(args):
    base = dict(dim=256, head=(64, 32), act="clip", king_buckets=args.king_buckets)
    return [{**base, "name": "halfka", "kind": "halfka"},
            {**base, "name": "hybrid_geom+halfka", "kind": "hybrid"},
            {**base, "name": "ultra_gating+betza", "kind": "ultra"}]


def exp_scaling(args):
    base = dict(kind="halfka", dim=256, head=(64, 32), act="clip",
                king_buckets=args.king_buckets)
    return [{**base, "name": "scale_800k", "budget": 800_000},
            {**base, "name": "scale_1600k", "budget": 1_600_000}]


def _count_armies(path, every=500, cap=400_000):
    """Sample the dataset (every Nth line) to rank armies by position count."""
    import collections
    c = collections.Counter(); seen = 0
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i % every:
                continue
            try:
                vm = json.loads(line).get("vm")
            except Exception:
                continue
            if vm is not None:
                c[vm] += 1; seen += 1
            if seen >= cap:
                break
    return c


def exp_crossarmy(args):
    """Compare HalfKA vs hybrid vs Ultra separately on each of the top-N armies,
    to check that the geometry-augmented models win across piece combinations,
    not just on the pooled data."""
    counts = _count_armies(args.data)
    top = [vm for vm, _ in counts.most_common(args.crossarmy_n)]
    cfgs = []
    for rank, vm in enumerate(top):
        for kind in ("halfka", "hybrid", "ultra"):
            cfgs.append({"name": f"army{rank+1}(vm{vm})_{kind}", "kind": kind,
                         "dim": 256, "head": (64, 32), "act": "clip",
                         "king_buckets": args.king_buckets, "vm_filter": [vm],
                         "budget": args.budget})
    return cfgs


def exp_champion(args):
    """Stack the three wins one at a time onto the hybrid: add the learnable
    ReLU, then the 64->64 head, then scale the transformer 256 -> 512 -> 1024."""
    kb = args.king_buckets
    base = dict(kind="hybrid", king_buckets=kb)
    return [
        {**base, "name": "champ_d256_clip_6432", "dim": 256, "head": (64, 32), "act": "clip"},
        {**base, "name": "champ_d256_learn_6432", "dim": 256, "head": (64, 32), "act": "learn"},
        {**base, "name": "champ_d256_learn_6464", "dim": 256, "head": (64, 64), "act": "learn"},
        {**base, "name": "champ_d512_learn_6464", "dim": 512, "head": (64, 64), "act": "learn"},
        {**base, "name": "champ_d1024_learn_6464", "dim": 1024, "head": (64, 64), "act": "learn"},
    ]


EXPS = {"arch": exp_arch, "relu": exp_relu, "models": exp_models,
        "scaling": exp_scaling, "crossarmy": exp_crossarmy, "champion": exp_champion}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=list(EXPS))
    ap.add_argument("--data", default="data/bigdb/bigdb.jsonl")
    ap.add_argument("--variants", default="data/bigdb/variants.json")
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--val-cap", type=int, default=30_000)
    ap.add_argument("--total-lines", type=int, default=134_420_914)
    ap.add_argument("--chunk-lines", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--king-buckets", type=int, default=16)
    ap.add_argument("--crossarmy-n", type=int, default=3,
                    help="number of top armies for the crossarmy experiment")
    ap.add_argument("--out", default="models/experiments")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    keep_awake()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    variants = json.load(open(args.variants, encoding="utf-8"))
    reg = BID.registry_from_variants(variants)
    val_keep = max(1, args.total_lines // max(1, args.val_cap))
    train_keep = max(1, args.total_lines // max(1, args.budget * args.epochs))
    outpath = os.path.join(args.out, f"{args.exp}.json")
    results = json.load(open(outpath)) if os.path.exists(outpath) else []
    done = {r["name"] for r in results}
    print(f"device={dev} exp={args.exp} types={reg.num_types} "
          f"val_stride={val_keep} train_stride={train_keep}", flush=True)

    cfgs = EXPS[args.exp](args)
    for cfg in cfgs:
        if cfg["name"] in done:
            print(f"skip {cfg['name']} (done)", flush=True)
            continue
        try:
            res = run_cfg(cfg, args, reg, variants, val_keep, train_keep, dev,
                          vm_filter=cfg.get("vm_filter"))
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"name": cfg["name"], "error": f"{type(e).__name__}: {e}"}
        results.append(res)
        json.dump(results, open(outpath, "w"), indent=2)

    ranked = sorted((r for r in results if "val_loss" in r), key=lambda r: r["val_loss"])
    print("\n" + "=" * 78)
    print(f"{'run':<20}{'val_loss':>10}{'params':>14}{'MB':>8}{'clip':>14}{'min':>7}")
    print("-" * 78)
    for r in ranked:
        clip = f"[{r['clip_lo']},{r['clip_hi']}]"
        print(f"{r['name']:<20}{r['val_loss']:>10.5f}{r['params']:>14,}"
              f"{r['nnue_mb']:>8}{clip:>14}{r['minutes']:>7.1f}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<20}   ERROR: {r['error']}")
    print("=" * 78)
    print(f"results -> {outpath}", flush=True)


if __name__ == "__main__":
    main()
