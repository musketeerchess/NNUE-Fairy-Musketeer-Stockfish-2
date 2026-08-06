"""
Parse the large database (Big Database.zip) into one streaming on-disk dataset.

Files are read game by game straight from the zip, so memory stays tiny no
matter how large a file is (some are 400+ MB). Every file is used. Only files
with inline evaluations produce records here; the rest need engine re-evaluation
(handled separately). Each PGN file has its own VariantMen, so we store a small
variant table and reference it by id per record, keeping the JSONL compact.

Robust: survives a bad game or file, saves the variant table periodically, and
stops cleanly if the disk gets low.

Output:
  data/bigdb/variants.json    list of unique VariantMen strings
  data/bigdb/bigdb.jsonl       {fen, score_cp, ply, result_stm, vm}

Usage:
    python scripts/parse_bigdb.py --zip "Big Database.zip" --out data/bigdb
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pgn_to_fen import parse_game, parse_score, result_to_stm   # noqa: E402
from musketeer import Board                                     # noqa: E402

EVAL_RE = re.compile(r'\{[+-]?[0-9]+\.[0-9]+/[0-9]')


def stream_games(binf):
    """Yield one PGN game (text) at a time from a binary line stream, so a huge
    file never gets fully loaded into memory."""
    cur = []
    for bline in binf:
        line = bline.decode("utf-8", "replace")
        if line.startswith("[Event ") and cur:
            yield "".join(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        yield "".join(cur)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="Big Database.zip")
    ap.add_argument("--out", default="data/bigdb")
    ap.add_argument("--progress-every", type=int, default=200000)
    ap.add_argument("--resume", action="store_true",
                    help="continue from checkpoint.json, appending to bigdb.jsonl")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    files = [n for n in z.namelist() if n.endswith(".pgn")]

    vpath = os.path.join(args.out, "variants.json")
    jpath = os.path.join(args.out, "bigdb.jsonl")
    cpath = os.path.join(args.out, "checkpoint.json")

    variants: dict[str, int] = {}
    vlist: list[str] = []
    start_file = 0
    if args.resume and os.path.exists(cpath):
        ck = json.load(open(cpath))
        start_file = ck["next_file"]
        vlist = json.load(open(vpath))
        variants = {v: i for i, v in enumerate(vlist)}
        print(f"RESUME from file index {start_file} "
              f"({ck.get('pos', 0):,} positions already saved, {len(vlist)} variants)",
              flush=True)
    jf = open(jpath, "a" if args.resume else "w", encoding="utf-8", buffering=1 << 20)

    st = {"pos": 0, "games": 0, "fail": 0, "files": 0, "noeval": 0, "stop": False}
    t0 = time.time()

    def progress():
        el = time.time() - t0
        free = shutil.disk_usage(".").free / 1e9
        print(f"  {st['pos']:,} positions | {st['files']} files | "
              f"{st['pos']/max(el,1):.0f} pos/s | {el/60:.1f} min | free {free:.1f}GB",
              flush=True)
        json.dump(vlist, open(vpath, "w"))
        jf.flush()
        if free < 3.0:
            print("!! LOW DISK, stopping cleanly", flush=True)
            st["stop"] = True

    for fi, name in enumerate(files):
        if st["stop"]:
            break
        if fi < start_file:
            continue
        try:
            with z.open(name) as binf:
                games = stream_games(binf)
                try:
                    first = next(games)
                except StopIteration:
                    continue
                if not EVAL_RE.search(first):       # peek: no evals -> skip file
                    st["noeval"] += 1
                    continue
                st["files"] += 1
                for chunk in itertools.chain([first], games):
                    g = parse_game(chunk)
                    if g is None or "VariantMen" not in g.headers or "FEN" not in g.headers:
                        continue
                    vm = g.headers["VariantMen"]
                    if vm not in variants:
                        variants[vm] = len(vlist); vlist.append(vm)
                    vid = variants[vm]
                    try:
                        board = Board.from_fen(g.headers["FEN"], vm)
                    except Exception:
                        st["fail"] += 1; continue
                    result = g.headers.get("Result", "*")
                    ply = 0
                    for san, comment in g.moves:
                        score = parse_score(comment) if comment else None
                        if score is not None:
                            jf.write(json.dumps({
                                "fen": board.to_fen(), "score_cp": score, "ply": ply,
                                "result_stm": result_to_stm(result, board.side),
                                "vm": vid}) + "\n")
                            st["pos"] += 1
                            if st["pos"] % args.progress_every == 0:
                                progress()
                                if st["stop"]:
                                    break
                        try:
                            board.apply_san(san)
                        except Exception:
                            st["fail"] += 1; break
                        ply += 1
                    st["games"] += 1
                    if st["stop"]:
                        break
        except Exception as e:
            print(f"  [error {name}: {type(e).__name__}] skipping", flush=True)
            continue
        # checkpoint after each fully-completed file (clean resume point)
        json.dump({"next_file": fi + 1, "pos": st["pos"]}, open(cpath, "w"))
        if (fi + 1) % 20 == 0:
            print(f"  ...{fi+1}/{len(files)} files scanned", flush=True)

    jf.close()
    json.dump(vlist, open(vpath, "w"))
    sz = os.path.getsize(jpath) / 1e9
    print(f"\nDONE: {st['files']} eval files ({st['noeval']} no-eval skipped), "
          f"{st['pos']:,} positions, {len(vlist)} variants, {st['fail']} game skips, "
          f"{sz:.1f} GB, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
