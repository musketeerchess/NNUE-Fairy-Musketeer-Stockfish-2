"""
Play a game of Musketeer against a trained net.

One-exchange mode (used to play via chat): apply the human's move, let the net
reply, then print the board, the net's move, and the human's legal moves.

    python play_human.py --net model3 --moves "e2e4 e7e5" --usermove g1f3
    python play_human.py --net model3 --moves ""            # show the start

The private engine supplies legal moves / FENs; the net (1-ply) chooses its
reply.  Human plays White by default.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

from arena import Arbiter, load_players     # noqa: E402
from musketeer import Board                 # noqa: E402

VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
START = ("*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** "
         "w KQkq - 0 1")


def show(arb, start, stack):
    fen, in_check = arb.board(start, stack)
    b = Board.from_fen(fen, VARIANT_MEN)
    print("\n" + b.ascii())
    print(f"\nFEN: {fen}")
    return fen, in_check


def terminal_msg(arb, start, stack):
    moves = arb.legal_moves(start, stack)
    if moves:
        return None, moves
    _, in_check = arb.board(start, stack)
    fen, _ = arb.board(start, stack)
    stm = fen.split()[1]
    if in_check:
        return f"CHECKMATE — {'Black' if stm=='w' else 'White'} wins!", []
    return "STALEMATE — draw.", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model3")
    ap.add_argument("--moves", default="")
    ap.add_argument("--usermove", default=None)
    ap.add_argument("--start", default=START)
    args = ap.parse_args()

    players = {p.name: p for p in load_players()}
    if args.net not in players:
        raise SystemExit(f"net '{args.net}' not found; have {list(players)}")
    player = players[args.net]
    arb = Arbiter()
    stack = args.moves.split() if args.moves.strip() else []

    # 1) apply the human's move (if any)
    if args.usermove:
        legal = arb.legal_moves(args.start, stack)
        if args.usermove not in legal:
            print(f"Illegal move '{args.usermove}'. Your legal moves are:\n"
                  f"  {' '.join(sorted(legal))}")
            arb.close(); return
        stack.append(args.usermove)
        print(f"You played: {args.usermove}")
        msg, _ = terminal_msg(arb, args.start, stack)
        if msg:
            show(arb, args.start, stack)
            print(f"\n{msg}\nmoves: {' '.join(stack)}")
            arb.close(); return

    # 2) net replies (if it is the net's turn)
    fen, _ = arb.board(args.start, stack)
    if fen.split()[1] == "b":                       # net = Black
        legal = arb.legal_moves(args.start, stack)
        if legal:
            reply = player.choose(arb, args.start, stack, legal)
            stack.append(reply)
            print(f"[{player.name}] replies: {reply}")

    # 3) show board + the human's options
    show(arb, args.start, stack)
    msg, your_moves = terminal_msg(arb, args.start, stack)
    if msg:
        print(f"\n{msg}")
    else:
        print(f"\nYour move — legal options ({len(your_moves)}):")
        print("  " + " ".join(sorted(your_moves)))
    print(f"\nmoves: {' '.join(stack)}")
    arb.close()


if __name__ == "__main__":
    main()
