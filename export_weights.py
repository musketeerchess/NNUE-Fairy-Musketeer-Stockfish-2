"""
Milestone 8 -- export a trained net to a portable, documented format.

Writes each layer's weights/biases to a single ``.npz`` plus an architecture
``.json`` header, so the trained evaluation can be integrated into any engine or
re-loaded without the Python class.  This is the format-independent hand-off; the
engine-native ``.nnue`` (loadable by a matching Fairy-Stockfish build) is
produced by the official ``variant-nnue-pytorch/serialize.py`` once the
architecture is fixed with the client -- see docs/Milestone8_Functional_Models.md.

Usage:
    python export_weights.py --ckpt models/model3.pt --out models/model3_export
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/model3.pt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.ckpt)[0] + "_export"

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]

    arrays = {}
    layers = []
    for name, tensor in sd.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        arrays[name] = arr
        layers.append({"name": name, "shape": list(arr.shape)})

    header = {
        "source_checkpoint": os.path.basename(args.ckpt),
        "arch": ck.get("arch", "unknown"),
        "input_features": int(ck.get("n_in", 128)),
        "width": int(ck.get("width", 0)) or None,
        "hidden_layers": int(ck.get("hidden", 0)),
        "eval_scale": float(ck.get("scale", 361.0)),
        "activation": "clipped_relu(0,1)",
        "note": ("Fully-connected NNUE. Evaluation is in side-to-move centipawns "
                 "after multiplying the network output; squash with "
                 "sigmoid(eval/eval_scale) for a win probability."),
        "layers": layers,
    }

    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w", encoding="utf-8") as fh:
        json.dump(header, fh, indent=2)
    print(f"wrote {out}.npz ({sum(a.size for a in arrays.values()):,} params) "
          f"and {out}.json")
    print("layers:")
    for l in layers:
        print(f"  {l['name']:20s} {l['shape']}")


if __name__ == "__main__":
    main()
