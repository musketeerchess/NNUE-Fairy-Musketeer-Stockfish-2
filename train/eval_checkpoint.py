"""Validate a saved checkpoint on the held-out 50k set (single-process, safe)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import torch
from torch.utils.data import DataLoader
import betza_id as BID
from parallel import EncodingIterable, COLLATE, SCALE
from compare_parallel import build_model, forward, _to_dev

ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/bigdb/hybrid_FULL.pt"
variants = json.load(open("data/bigdb/variants.json", encoding="utf-8"))
reg = BID.registry_from_variants(variants)
spec = {"kind": "hybrid", "name": "hybrid", "king_buckets": 16}
model = build_model(spec, reg, 256, "cpu")
ck = torch.load(ckpt, map_location="cpu")
model.load_state_dict(ck["model"]); model.eval()
val_keep = 134420914 // 50000
ds = EncodingIterable("data/bigdb/bigdb.jsonl", variants, reg, spec, val_keep, 1,
                      chunk_lines=100000, seed=99, mode="val")
dl = DataLoader(ds, batch_size=8192, num_workers=0, collate_fn=COLLATE["hybrid"])
t0 = time.time(); tot = 0.0; n = 0
with torch.no_grad():
    for inp, S in dl:
        q = forward(spec, model, inp)
        tot += ((torch.sigmoid(q / SCALE) - S) ** 2).sum().item(); n += S.numel()
print(f"FINAL validation loss = {tot/max(1,n):.5f}  on {n:,} held-out positions "
      f"({(time.time()-t0)/60:.1f} min)", flush=True)
