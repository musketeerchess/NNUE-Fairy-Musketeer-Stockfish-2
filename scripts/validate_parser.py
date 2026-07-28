"""
Cross-check the parser against the private engine.

Samples FENs produced by the parser and asks the engine to search each one; a
legal, well-formed Musketeer position yields a real ``bestmove`` (not
``(none)``) and no parse error.  If our parser had corrupted a position
(mis-applied gating, dropped a piece, wrong side to move, ...) the position
would usually become illegal and the engine would reject it or return
``bestmove (none)``.

Usage:
    python scripts/validate_parser.py data/processed/train1.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

ENGINE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "engine",
    "fairy-stockfish-largeboards_x86-64.exe"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    fens = [json.loads(l)["fen"] for l in open(args.jsonl, encoding="utf-8")]
    if not fens:
        print("no records"); return
    step = max(1, len(fens) // args.n)
    sample = fens[::step][: args.n]

    cmds = ["setoption name UCI_Variant value musketeer"]
    for fen in sample:
        cmds.append(f"position fen {fen}")
        cmds.append("go depth 2")
    cmds.append("quit")
    proc = subprocess.run([ENGINE], input="\n".join(cmds) + "\n",
                          capture_output=True, text=True, timeout=600)
    out_lines = proc.stdout.splitlines()
    bestmoves = [l.split()[1] for l in out_lines if l.startswith("bestmove")]

    n = len(sample)
    got = len(bestmoves)
    none = sum(1 for b in bestmoves if b == "(none)")
    legal = got - none
    print(f"sampled positions : {n}")
    print(f"engine replies     : {got}")
    print(f"legal (real move)  : {legal}  ({100*legal/max(1,n):.1f}%)")
    print(f"bestmove (none)    : {none}")
    if got < n:
        print("WARNING: fewer replies than positions -> engine rejected some FENs")
    if "info string" in proc.stdout.lower() and "error" in proc.stdout.lower():
        print("engine emitted error strings; check FENs")
    # show any position the engine could not move in
    if none:
        idx = [i for i, b in enumerate(bestmoves) if b == "(none)"][:5]
        for i in idx:
            print("  no-move FEN:", sample[i])


if __name__ == "__main__":
    main()
