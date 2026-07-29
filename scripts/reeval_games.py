"""
Re-evaluate result-only PGNs with the Musketeer Stockfish engine.

Newer game PGNs contain no inline evaluations, and their pieces are custom
(redefined in the `VariantMen` header). This tool:

  1. builds a `variants.ini` from the PGN's `VariantMen` (so the engine uses the
     *exact* tournament rules -- essential, because the engine's built-in rules
     differ, e.g. it would move `I` like a knight instead of ZD);
  2. loads it into the engine via `VariantPath`;
  3. replays each game (our own board) and asks the engine to evaluate every
     position;
  4. emits `(fen, score_cp, move, ply, result_stm)` records -- the same schema
     as `pgn_to_fen.py`, ready for NNUE training.

Usage:
    python scripts/reeval_games.py "data/games_new/Games/ii vs jj-1.pgn" \
        --engine engine_v2/musketeer-stockfish_x86-64.exe --depth 10 \
        --max-games 50 --jsonl data/processed/reeval_ii_jj.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "third_party", "chess-variant-stats"))
from pgn_to_fen import split_games, parse_game, result_to_stm   # noqa: E402
from musketeer import Board, WHITE                              # noqa: E402
import uci                                                      # noqa: E402

MATE_CP = 30000


def win_path(p: str) -> str:
    """Absolute Windows-style path the engine can open."""
    return os.path.abspath(p).replace("\\", "/")


def make_ini(variant_men: str, start_fen: str, path: str, section: str = "musketeer"):
    lines = [f"[{section}]", f"startFen = {start_fen}"]
    n = 0
    for chunk in variant_men.split(";"):
        if ":" not in chunk:
            continue
        sym, betza = chunk.split(":", 1)
        n += 1
        lines.append(f"customPiece{n} = {sym.strip()}:{betza.strip()}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return n


def score_cp(infos) -> int | None:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--engine", default="engine_v2/musketeer-stockfish_x86-64.exe")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--ini", default="data/games_new/_reeval_rules.ini")
    args = ap.parse_args()

    text = open(args.pgn, encoding="utf-8", errors="replace").read()
    games = [g for g in (parse_game(c) for c in split_games(text))
             if g and "VariantMen" in g.headers and "FEN" in g.headers]
    if not games:
        raise SystemExit("no games with VariantMen/FEN")
    variant_men = games[0].headers["VariantMen"]

    # one ini for the file (shared VariantMen); startFen from the first game
    npieces = make_ini(variant_men, games[0].headers["FEN"], args.ini)
    print(f"ini: {npieces} custom pieces -> {args.ini}", flush=True)

    def make_engine():
        e = uci.Engine([os.path.abspath(args.engine)])
        e.setoption("VariantPath", win_path(args.ini))
        e.setoption("UCI_Variant", "musketeer")
        return e

    engine = make_engine()
    restarts = [0]

    def evaluate(fen):
        """Engine eval of one FEN; restarts the engine on a pipe/process error."""
        nonlocal engine
        try:
            engine.position(fen, [])
            _, infos = engine.go(depth=args.depth)
            return score_cp(infos)
        except Exception:                       # engine died / pipe broke
            try:
                engine.process.kill()
            except Exception:
                pass
            engine = make_engine()
            restarts[0] += 1
            try:                                 # one retry after restart
                engine.position(fen, [])
                _, infos = engine.go(depth=args.depth)
                return score_cp(infos)
            except Exception:
                return None                      # skip this position

    jf = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    n_games = n_pos = n_fail = 0
    limit = args.max_games or len(games)
    for gi, g in enumerate(games[:limit]):
        try:
            board = Board.from_fen(g.headers["FEN"], variant_men)
        except Exception:
            n_fail += 1; continue
        result = g.headers.get("Result", "*")
        ply = 0
        ok = True
        for san, _ in g.moves:
            fen = board.to_fen()
            stm = board.side
            sc = evaluate(fen)
            if sc is not None:
                rec = {"fen": fen, "score_cp": sc, "move": san, "ply": ply,
                       "stm": 1 if stm == WHITE else -1,
                       "result_stm": result_to_stm(result, stm)}
                if jf:
                    jf.write(json.dumps(rec) + "\n")
                n_pos += 1
            try:
                board.apply_san(san)
            except Exception:
                ok = False; break
            ply += 1
        n_games += 1
        if (gi + 1) % 10 == 0:
            print(f"  {gi+1}/{limit} games, {n_pos} positions", flush=True)
    if jf:
        jf.close()
    engine.write("quit\n")
    print(f"\ndone: {n_games} games, {n_pos} eval'd positions, {n_fail} skipped, "
          f"{restarts[0]} engine restarts", flush=True)


if __name__ == "__main__":
    main()
