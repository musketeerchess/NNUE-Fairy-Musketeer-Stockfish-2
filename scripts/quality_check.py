"""
Database quality check (Zied's spec).

Flags games whose result contradicts a large engine advantage: a side is clearly
winning by the engine's evaluation, yet the game is drawn or lost by that side.
Those games teach the NNUE a false label and are worth pulling out of training.

For each game we:
  1. replay it on our own board and collect the position after every move;
  2. ask the engine (at a deep search) for the evaluation of the final position,
     converted to White's point of view;
  3. if |eval| is large but the result does not match, walk backwards over the
     last few positions, evaluating each at the same depth, to show whether the
     advantage was real and squandered or just a late swing;
  4. classify the likely cause:
       - "insufficient_material": the winning side has only its king plus a single
         non-royal piece against a bare king, which is often a theoretical draw
         (e.g. K+Unicorn vs K) -- an expected draw, not a bad game;
       - "advantage_not_converted": a clear, sustained advantage that ended in a
         draw or loss -- the games most likely hurt by time control or technique.

A first version: it triages games for review; it does not itself decide the true
result. Needs the engine and the per-file VariantMen (built into a variants.ini).

Usage:
    python scripts/quality_check.py "uni hawk training nnue-1.pgn" \
        --engine engine/fairy-stockfish-largeboards_x86-64.exe \
        --depth 20 --threshold 400 --max-games 200 --out quality_report.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "third_party", "chess-variant-stats"))
from pgn_to_fen import split_games, parse_game                # noqa: E402
from musketeer import Board, WHITE, BLACK                     # noqa: E402
import uci                                                    # noqa: E402
from reeval_games import make_ini, win_path, score_cp         # noqa: E402


def result_white_pov(result: str):
    """1.0 White win, 0.0 Black win, 0.5 draw, None if unknown."""
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)


def winning_side_material(board: Board):
    """(non-royal pieces of White, of Black) as letter lists, and bare-king flags."""
    w, b = [], []
    for (f, r), (letter, color) in board.board.items():
        pc = board.pieces.get(letter)
        if pc is not None and pc.is_royal:
            continue
        (w if color == WHITE else b).append(letter)
    return w, b


def classify(board, adv_side):
    """adv_side: WHITE/BLACK who the engine says is winning."""
    w, b = winning_side_material(board)
    strong = w if adv_side == WHITE else b
    weak = b if adv_side == WHITE else w
    if len(weak) == 0 and len(strong) <= 1:
        # king + at most one piece vs bare king -> often a theoretical draw
        return "insufficient_material"
    return "advantage_not_converted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--engine", default="engine/fairy-stockfish-largeboards_x86-64.exe")
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--threshold", type=int, default=400,
                    help="centipawn advantage that should have won")
    ap.add_argument("--lookback", type=int, default=8,
                    help="positions from the end to evaluate for the trajectory")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--ini", default="data/_quality_rules.ini")
    ap.add_argument("--out", default="quality_report.csv")
    args = ap.parse_args()

    text = open(args.pgn, encoding="utf-8", errors="replace").read()
    games = [g for g in (parse_game(c) for c in split_games(text))
             if g and "VariantMen" in g.headers and "FEN" in g.headers]
    if not games:
        raise SystemExit("no games with VariantMen/FEN")
    variant_men = games[0].headers["VariantMen"]
    os.makedirs(os.path.dirname(args.ini) or ".", exist_ok=True)
    make_ini(variant_men, games[0].headers["FEN"], args.ini)

    def make_engine():
        e = uci.Engine([os.path.abspath(args.engine)])
        e.setoption("VariantPath", win_path(args.ini))
        e.setoption("UCI_Variant", "musketeer")
        return e

    engine = make_engine()

    def evaluate(fen):
        nonlocal engine
        for attempt in (0, 1):
            try:
                engine.position(fen, [])
                _, infos = engine.go(depth=args.depth)
                return score_cp(infos)
            except Exception:
                try:
                    engine.process.kill()
                except Exception:
                    pass
                engine = make_engine()
        return None

    flagged = []
    limit = args.max_games or len(games)
    for gi, g in enumerate(games[:limit]):
        res = result_white_pov(g.headers.get("Result", "*"))
        if res is None:
            continue
        try:
            board = Board.from_fen(g.headers["FEN"], variant_men)
        except Exception:
            continue
        fens = []
        ok = True
        for san, _ in g.moves:
            try:
                board.apply_san(san)
            except Exception:
                ok = False
                break
            fens.append(board.to_fen())
        if not ok or not fens:
            continue

        final_board = board
        stm = final_board.side
        ev = evaluate(fens[-1])
        if ev is None:
            continue
        # engine eval is from side-to-move; convert to White's point of view
        wpov = ev if stm == WHITE else -ev
        if abs(wpov) < args.threshold:
            continue                              # no large advantage -> fine
        adv_side = WHITE if wpov > 0 else BLACK
        adv_won = (res == 1.0 and adv_side == WHITE) or (res == 0.0 and adv_side == BLACK)
        if adv_won:
            continue                              # advantage converted -> fine

        # disparity: big advantage but not a win for that side -> flag + trajectory
        traj = []
        for fen in fens[-args.lookback:]:
            e = evaluate(fen)
            if e is not None:
                b2 = Board.from_fen(fen, variant_men)
                traj.append(e if b2.side == WHITE else -e)
        cause = classify(final_board, adv_side)
        w, b = winning_side_material(final_board)
        flagged.append({
            "game": gi + 1,
            "result": g.headers.get("Result", "*"),
            "white_pov_eval": wpov,
            "advantage_side": "White" if adv_side == WHITE else "Black",
            "cause": cause,
            "white_pieces": "".join(sorted(w)),
            "black_pieces": "".join(sorted(b)),
            "final_fen": fens[-1],
            "trajectory": " ".join(str(t) for t in traj),
        })
        print(f"  game {gi+1}: {g.headers.get('Result')} but "
              f"{'White' if adv_side==WHITE else 'Black'} +{abs(wpov)}cp -> {cause}",
              flush=True)

    fields = ["game", "result", "white_pov_eval", "advantage_side", "cause",
              "white_pieces", "black_pieces", "trajectory", "final_fen"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for row in flagged:
            wr.writerow(row)
    n_ins = sum(1 for r in flagged if r["cause"] == "insufficient_material")
    n_bad = len(flagged) - n_ins
    print(f"\nchecked {min(limit,len(games))} games | flagged {len(flagged)}: "
          f"{n_bad} advantage-not-converted (likely poor quality), "
          f"{n_ins} insufficient-material (expected draws) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
