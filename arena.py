"""
Milestone 6 -- self-play arena: determine the best NNUE and delete bad nets.

The four trained nets play a round-robin against each other and are ranked by
score; the weakest are culled.  Because our nets are custom PyTorch (not engine
`.nnue`), we let the **private engine act purely as the game arbiter** -- it
generates the legal Musketeer moves and adjudicates checkmate/stalemate, which
it does perfectly -- while each **net is the brain**: at every move the net
evaluates the position after each legal reply and plays the one that minimises
the opponent's advantage (1-ply negamax on the net's evaluation).  No `.nnue`
export or GPU is required.

Culling ("delete bad nets") moves the losers to ``models/discarded/`` rather
than hard-deleting, so nothing is lost irreversibly.

Usage:
    python arena.py --games 4 --book data/processed/musketeer_book.epd \
        --move-cap 120 --discard-below 0.30
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))
from features import encode_fen, encode_fen_model3, gating_finished  # noqa: E402
from musketeer import Board                                          # noqa: E402
from model1 import Model1                                            # noqa: E402
from model2 import Model2                                            # noqa: E402
from model3 import Model3                                            # noqa: E402
from model4 import Model4                                            # noqa: E402

ENGINE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "engine", "fairy-stockfish-largeboards_x86-64.exe"))
VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")


# --------------------------------------------------------------------------- #
# Engine arbiter (move generation + FENs + terminal detection only)
# --------------------------------------------------------------------------- #
class Arbiter:
    def __init__(self, variant: str = "musketeer"):
        self.p = subprocess.Popen([ENGINE], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, universal_newlines=True,
                                  bufsize=1)
        self._w("uci"); self._read_until("uciok")
        self._w(f"setoption name UCI_Variant value {variant}")

    def _w(self, s: str) -> None:
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        out = []
        while True:
            line = self.p.stdout.readline()
            if not line and self.p.poll() is not None:
                break
            out.append(line.rstrip("\n"))
            if line.startswith(token):
                break
        return out

    def _pos(self, start: str, stack: list[str]) -> str:
        mv = f" moves {' '.join(stack)}" if stack else ""
        return f"position fen {start}{mv}"

    def legal_moves(self, start: str, stack: list[str]) -> list[str]:
        self._w(self._pos(start, stack))
        self._w("go perft 1")
        moves = []
        for line in self._read_until("Nodes searched"):
            if len(line) > 2 and line[0].isalpha() and ":" in line and line[1].isdigit():
                moves.append(line.split(":")[0].strip())
        return moves

    def board(self, start: str, stack: list[str]) -> tuple[str, bool]:
        self._w(self._pos(start, stack)); self._w("d"); self._w("isready")
        fen, chk = None, False
        for line in self._read_until("readyok"):
            if line.startswith("Fen:"):
                fen = line.split(":", 1)[1].strip()
            elif line.startswith("Checkers:"):
                chk = bool(line.split(":", 1)[1].strip())
        return fen, chk

    def child_fens(self, start: str, stack: list[str], moves: list[str]) -> list[str]:
        # One round-trip per move: `d` output always ends with a "Checkers:"
        # line, so read up to it immediately (avoids a stdout-buffer deadlock).
        fens: list[str] = []
        for m in moves:
            self._w(self._pos(start, stack + [m])); self._w("d")
            fen = None
            for ln in self._read_until("Checkers:"):
                if ln.startswith("Fen:"):
                    fen = ln.split(":", 1)[1].strip()
            fens.append(fen)
        return fens

    def close(self):
        try:
            self._w("quit"); self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


# --------------------------------------------------------------------------- #
# Net players
# --------------------------------------------------------------------------- #
class NetPlayer:
    def __init__(self, name: str, model, encoder, gating: bool = False):
        self.name = name
        self.model = model.eval()
        self.encoder = encoder
        self.gating = gating

    def _eval(self, fen: str) -> float:
        x = torch.from_numpy(self.encoder(fen, VARIANT_MEN)).unsqueeze(0)
        with torch.no_grad():
            if self.gating:
                if gating_finished(Board.from_fen(fen, VARIANT_MEN)):
                    q = self.model.forward_post(x[:, :64])
                else:
                    q = self.model.forward_pre(x)
            else:
                q = self.model(x)
        return float(q.item())

    def choose(self, arb: Arbiter, start: str, stack: list[str], moves: list[str]) -> str:
        # child fen's side-to-move is the opponent; minimise their eval.
        childs = arb.child_fens(start, stack, moves)
        best, best_mv = float("inf"), moves[0]
        for mv, cf in zip(moves, childs):
            if cf is None:
                continue
            e = self._eval(cf)
            if e < best:
                best, best_mv = e, mv
        return best_mv


def load_players() -> list[NetPlayer]:
    players = []
    # Prefer the fully-trained checkpoints; fall back to the smoke ones.
    reg = [
        ("models/model1.pt", "models/model1_smoke.pt", Model1, encode_fen, False),
        ("models/model2.pt", "models/model2_smoke.pt", Model2, encode_fen, True),
        ("models/model3.pt", "models/model3_smoke.pt", Model3, encode_fen_model3, False),
        ("models/model4.pt", "models/model4_smoke.pt", Model4, encode_fen, False),
    ]
    for full, smoke, cls, enc, gating in reg:
        path = full if os.path.exists(full) else smoke
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if cls is Model2:
            model = cls(hidden=ck.get("hidden", 1))
        else:
            model = cls(width=ck.get("width", 256), hidden=ck.get("hidden", 2))
        model.load_state_dict(ck["model"])
        name = os.path.basename(path).replace("_smoke.pt", "").replace(".pt", "")
        players.append(NetPlayer(name, model, enc, gating))
    return players


# --------------------------------------------------------------------------- #
# Play one game -> result from white's perspective (+1 / 0 / -1)
# --------------------------------------------------------------------------- #
def play_game(arb: Arbiter, white: NetPlayer, black: NetPlayer,
              start: str, move_cap: int) -> int:
    stack: list[str] = []
    for _ in range(move_cap):
        moves = arb.legal_moves(start, stack)
        if not moves:
            _, in_check = arb.board(start, stack)
            fen, _ = arb.board(start, stack)
            stm = fen.split()[1]
            if in_check:
                return -1 if stm == "w" else 1     # side to move is checkmated
            return 0                                # stalemate
        fen, _ = arb.board(start, stack)
        mover = white if fen.split()[1] == "w" else black
        stack.append(mover.choose(arb, start, stack, moves))
    return 0                                         # move cap -> draw


# --------------------------------------------------------------------------- #
# Round-robin
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4, help="games per ordered pair")
    ap.add_argument("--book", default="data/processed/musketeer_book.epd")
    ap.add_argument("--move-cap", type=int, default=120)
    ap.add_argument("--discard-below", type=float, default=0.30,
                    help="cull nets scoring below this fraction of max")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    book = [l.strip() for l in open(args.book, encoding="utf-8") if l.strip()]
    players = load_players()
    if len(players) < 2:
        raise SystemExit("need >=2 trained nets in models/*.pt")
    print("Players:", ", ".join(p.name for p in players))

    arb = Arbiter()
    score = {p.name: 0.0 for p in players}
    played = {p.name: 0 for p in players}
    print("\n=== games ===")
    for i, a in enumerate(players):
        for j, b in enumerate(players):
            if i >= j:
                continue
            for g in range(args.games):
                start = book[(i * 7 + j * 3 + g) % len(book)]
                w, bl = (a, b) if g % 2 == 0 else (b, a)   # alternate colours
                r = play_game(arb, w, bl, start, args.move_cap)
                sw = 0.5 + 0.5 * r          # white score
                score[w.name] += sw; score[bl.name] += (1 - sw)
                played[w.name] += 1; played[bl.name] += 1
                res = {1: "1-0", 0: "1/2", -1: "0-1"}[r]
                print(f"  {w.name} (W) vs {bl.name} (B): {res}")
    arb.close()

    print("\n=== standings (points / games) ===")
    ranked = sorted(players, key=lambda p: score[p.name], reverse=True)
    mx = max(score.values()) or 1.0
    for rank, p in enumerate(ranked, 1):
        pct = score[p.name] / max(1, played[p.name])
        print(f"  {rank}. {p.name:8s} {score[p.name]:.1f}/{played[p.name]}"
              f"  ({pct*100:.0f}%)")

    best = ranked[0].name
    print(f"\nBEST NET: {best}")

    # delete (archive) bad nets
    os.makedirs("models/discarded", exist_ok=True)
    culled = []
    for p in players:
        frac = score[p.name] / mx
        if p.name != best and frac < args.discard_below:
            for f in glob.glob(f"models/{p.name}*.pt"):
                dst = os.path.join("models/discarded", os.path.basename(f))
                os.replace(f, dst)
                culled.append(os.path.basename(f))
    if culled:
        print("culled (moved to models/discarded/):", ", ".join(culled))
    else:
        print("no nets below the discard threshold "
              f"({args.discard_below:.0%} of best) -- none culled")


if __name__ == "__main__":
    main()
