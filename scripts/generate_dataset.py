"""
Milestone 7 -- automate creation of the pre-configured NNUE database.

Follows the approach of `chess-variant-stats/generate_games.py` (engine
self-play from an opening book -> positions), but drives the **private Musketeer
engine directly** because the Musketeer variant is not available in public
pyffish.  Crucially, unlike chess-variant-stats it also records the engine's
**evaluation** for every position -- which is what NNUE training needs -- and
writes the exact same `(fen, score_cp, move, result)` schema as
`src/pgn_to_fen.py`, so generated data is a drop-in for the trainers.

This removes the dependency on hand-provided PGNs: to build a fresh database for
any piece combination, point it at an opening book and let the engine play.

Usage:
    python scripts/generate_dataset.py --count 20 --movetime 100 \
        --book data/processed/musketeer_book.epd \
        --jsonl data/processed/selfplay.jsonl --plain data/processed/selfplay.plain
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "third_party", "chess-variant-stats"))
import uci  # noqa: E402  (from chess-variant-stats)

ENGINE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "engine",
    "fairy-stockfish-largeboards_x86-64.exe"))
MATE_CP = 30000
MOVE_CAP = 300


def _drain_ready(engine: "uci.Engine") -> list[str]:
    """Send 'd' + isready and return the captured lines (board / Fen / Checkers)."""
    engine.write("d\n")
    engine.write("isready\n")
    return engine.read("readyok")


def board_info(engine: "uci.Engine") -> tuple[str, bool]:
    """Return (fen, in_check) for the current engine position."""
    fen, in_check = None, False
    for line in _drain_ready(engine):
        if line.startswith("Fen:"):
            fen = line.split(":", 1)[1].strip()
        elif line.startswith("Checkers:"):
            in_check = bool(line.split(":", 1)[1].strip())
    return fen, in_check


def score_cp(infos) -> int | None:
    """Extract side-to-move centipawns from the last depth's first pv."""
    if not infos:
        return None
    pv = infos[-1][0]
    sc = pv.get("score")
    if not sc:
        return None
    kind, val = sc[0], int(sc[1])
    if kind == "mate":
        return (MATE_CP - abs(val)) * (1 if val > 0 else -1)
    return val


def play_game(engine, variant, start_fen, movetime):
    """Self-play one game; yield (fen_before, score, move) and finally result."""
    engine.newgame()
    move_stack: list[str] = []
    records = []
    result_white = 0
    while len(move_stack) < MOVE_CAP:
        engine.position(start_fen, move_stack)
        fen_before, in_check = board_info(engine)
        bestmove, infos = engine.go(movetime=movetime)
        if bestmove in (None, "(none)", "0000"):
            # terminal: checkmate -> side to move loses; else draw (stalemate)
            stm = fen_before.split()[1] if fen_before else "w"
            if in_check:
                result_white = -1 if stm == "w" else 1
            else:
                result_white = 0
            break
        sc = score_cp(infos)
        if sc is not None and fen_before is not None:
            records.append((fen_before, sc, bestmove))
        move_stack.append(bestmove)
    else:
        # hit move cap -> adjudicate by last eval sign (from stm perspective)
        result_white = 0
    return records, result_white, len(move_stack)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=ENGINE)
    ap.add_argument("--variant", default="musketeer")
    ap.add_argument("--book", default="data/processed/musketeer_book.epd")
    ap.add_argument("--count", type=int, default=10, help="# games")
    ap.add_argument("--movetime", type=int, default=100, help="ms per move")
    ap.add_argument("--jsonl", default="data/processed/selfplay.jsonl")
    ap.add_argument("--plain", default="data/processed/selfplay.plain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    startfens = [l.strip() for l in open(args.book, encoding="utf-8") if l.strip()]
    engine = uci.Engine([args.engine])
    engine.setoption("UCI_Variant", args.variant)

    jf = open(args.jsonl, "w", encoding="utf-8")
    pf = open(args.plain, "w", encoding="utf-8")
    total = 0
    for gi in range(args.count):
        start = random.choice(startfens)
        records, result_white, plies = play_game(engine, args.variant, start, args.movetime)
        for ply, (fen, sc, mv) in enumerate(records):
            stm = fen.split()[1]
            rw = result_white
            result_stm = rw if stm == "w" else -rw
            rec = {"fen": fen, "score_cp": sc, "move": mv, "ply": ply,
                   "stm": 1 if stm == "w" else -1, "result_stm": result_stm}
            jf.write(json.dumps(rec) + "\n")
            pf.write(f"fen {fen}\nmove {mv}\nscore {sc}\nply {ply}\n"
                     f"result {result_stm}\ne\n")
            total += 1
        print(f"game {gi+1}/{args.count}: {plies} plies, "
              f"result(white)={result_white}, {len(records)} positions")
    jf.close(); pf.close()
    print(f"\nTOTAL {total} positions -> {args.jsonl}")
    engine.write("quit\n")


if __name__ == "__main__":
    main()
