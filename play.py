"""
Milestone 8 -- a *functional* Musketeer model.

Turns a trained net into a working move-picker: given any Musketeer position it
returns the move the net would play (1-ply search on the net's evaluation, with
the private engine supplying legal moves and resulting positions).  This is the
"functional model" in the directly-usable sense -- it plays legal Musketeer
chess right now, no engine recompilation needed.

Examples:
    # best move for a position, using the strongest net (model3)
    python play.py --fen "*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** w KQkq - 0 1"

    # let the net play a full game against itself and print the moves
    python play.py --selfplay --net model3 --move-cap 120
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

from arena import Arbiter, load_players     # noqa: E402

START = ("*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** "
         "w KQkq - 0 1")


def get_player(name: str):
    players = {p.name: p for p in load_players()}
    if name not in players:
        raise SystemExit(f"net '{name}' not found; have: {list(players)}")
    return players[name]


def best_move(arb: Arbiter, player, fen: str) -> str | None:
    moves = arb.legal_moves(fen, [])
    if not moves:
        return None
    return player.choose(arb, fen, [], moves)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fen", default=START)
    ap.add_argument("--net", default="model3", help="which trained net to use")
    ap.add_argument("--selfplay", action="store_true")
    ap.add_argument("--move-cap", type=int, default=120)
    args = ap.parse_args()

    player = get_player(args.net)
    arb = Arbiter()

    if not args.selfplay:
        mv = best_move(arb, player, args.fen)
        print(f"[{player.name}] best move: {mv}")
        arb.close()
        return

    # self-play: the net plays both sides
    print(f"[{player.name}] self-play from start:\n")
    stack: list[str] = []
    for ply in range(args.move_cap):
        moves = arb.legal_moves(args.fen, stack)
        if not moves:
            _, chk = arb.board(args.fen, stack)
            fen_now, _ = arb.board(args.fen, stack)
            stm = fen_now.split()[1]
            outcome = ("checkmate -- " + ("black" if stm == "w" else "white") + " wins") \
                if chk else "stalemate -- draw"
            print(f"\nGame over: {outcome}  ({len(stack)} plies)")
            break
        mv = player.choose(arb, args.fen, stack, moves)
        stack.append(mv)
        sep = "  " if ply % 2 == 0 else "\n"
        print(f"{ply//2+1 if ply%2==0 else '':>3}{'.' if ply%2==0 else '  '} {mv}",
              end=sep, flush=True)
    else:
        print(f"\nReached move cap ({args.move_cap} plies) -- adjudicated draw")
    arb.close()


if __name__ == "__main__":
    main()
