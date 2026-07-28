"""
Parse Musketeer PGNs into NNUE training records.

For every move in every game we emit the position *before* the move together
with the engine evaluation attached to that move, the move itself, the ply, and
the final game result:

    {fen, score_cp, move, ply, stm, result_stm}

  * ``score_cp``   – centipawns from the side-to-move's perspective (the
                     ``{+1.20/18}`` comment; +ve = good for the mover).
  * ``result_stm`` – game result from the side-to-move's perspective
                     (+1 win / 0 draw / -1 loss).  The PGN's "False draw claim"
                     comments are ignored; we trust the header ``Result``.

Output: JSONL (our format) and Stockfish/Fairy ``.plain`` text (trainer format).

Usage:
    python src/pgn_to_fen.py "data/raw/uni hawk training nnue-1.pgn" \
        --jsonl data/processed/train1.jsonl --plain data/processed/train1.plain
    python src/pgn_to_fen.py <pgn> --check 50   # replay-only, report failures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

from musketeer import Board, WHITE

# --------------------------------------------------------------------------- #
# PGN tokenising
# --------------------------------------------------------------------------- #
HEADER_RE = re.compile(r'\[(\w+)\s+"(.*)"\]')
# a move token: SAN (incl. gating "/X", promotion "=X", castling), no spaces
MOVE_RE = re.compile(r'^(O-O(?:-O)?|[A-Za-z][A-Za-z0-9=\-/]*)[+#]?[!?]*$')
SCORE_RE = re.compile(r'([+-]?)(?:M(\d+)|(\d+)\.(\d+)|(\d+))')

MATE_CP = 30000


@dataclass
class Game:
    headers: dict[str, str]
    moves: list[tuple[str, str | None]]   # (san, eval_comment_or_None)


def split_games(text: str):
    """Yield raw (header_block, movetext) pairs from a multi-game PGN."""
    # Games are separated by a blank line between the movetext of one game and
    # the '[Event' of the next.  Split on the '[Event' boundary.
    chunks = re.split(r'\n(?=\[Event )', text)
    for ch in chunks:
        ch = ch.strip()
        if ch:
            yield ch


def parse_game(chunk: str) -> Game | None:
    headers: dict[str, str] = {}
    lines = chunk.splitlines()
    i = 0
    for i, line in enumerate(lines):
        m = HEADER_RE.match(line.strip())
        if m:
            headers[m.group(1)] = m.group(2)
        elif line.strip() == "" and headers:
            break
    movetext = "\n".join(lines[i:])

    # strip the pre-move ASCII diagram block {-------- ... --------}
    # and keep eval comments.  We tokenise by walking the string.
    moves: list[tuple[str, str | None]] = []
    tokens = _tokenise_movetext(movetext)
    pending_move: str | None = None
    for kind, val in tokens:
        if kind == "move":
            if pending_move is not None:
                moves.append((pending_move, None))
            pending_move = val
        elif kind == "comment":
            if pending_move is not None:
                moves.append((pending_move, val))
                pending_move = None
        elif kind == "result":
            if pending_move is not None:
                moves.append((pending_move, None))
                pending_move = None
    if pending_move is not None:
        moves.append((pending_move, None))
    return Game(headers, moves)


def _tokenise_movetext(text: str):
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "{":
            j = text.find("}", i)
            if j == -1:
                break
            yield ("comment", text[i + 1:j])
            i = j + 1
        elif c == "(":                       # variation – skip (balanced)
            depth = 1; i += 1
            while i < n and depth:
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                i += 1
        elif c.isdigit() and _looks_like_move_number(text, i):
            while i < n and (text[i].isdigit() or text[i] == "."):
                i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in "{}()":
                j += 1
            tok = text[i:j]
            i = j
            if tok in ("1-0", "0-1", "1/2-1/2", "*"):
                yield ("result", tok)
            elif MOVE_RE.match(tok) and not tok[0].isdigit():
                yield ("move", tok)
            # else: NAG like $1, stray token -> ignore


def _looks_like_move_number(text: str, i: int) -> bool:
    j = i
    while j < len(text) and text[j].isdigit():
        j += 1
    return j < len(text) and text[j] == "."


def parse_score(comment: str) -> int | None:
    """Extract centipawns from a comment like '+1.20/18' or '+M5/12 0.3'."""
    if not comment:
        return None
    tok = comment.strip().split()[0]
    tok = tok.split("/")[0]
    m = SCORE_RE.match(tok)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    if m.group(2):                    # mate score  Mn
        return sign * (MATE_CP - int(m.group(2)))
    if m.group(3) is not None:        # d.dd  pawns
        return sign * (int(m.group(3)) * 100 + int(m.group(4).ljust(2, "0")[:2]))
    if m.group(5) is not None:        # integer pawns
        return sign * int(m.group(5)) * 100
    return None


def result_to_stm(result: str, stm: int) -> int | None:
    """Game result from the perspective of the side to move (stm)."""
    if result == "1-0":
        white = 1
    elif result == "0-1":
        white = -1
    elif result == "1/2-1/2":
        return 0
    else:
        return None
    return white if stm == WHITE else -white


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def iter_records(pgn_path: str, stop_on_error: bool = False):
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    n_games = n_ok = n_fail = n_pos = 0
    failures = []
    for chunk in split_games(text):
        g = parse_game(chunk)
        if g is None or "FEN" not in g.headers or "VariantMen" not in g.headers:
            continue
        n_games += 1
        try:
            board = Board.from_fen(g.headers["FEN"], g.headers["VariantMen"])
        except Exception as e:                       # noqa: BLE001
            n_fail += 1
            failures.append((n_games, "from_fen", str(e)))
            continue
        result = g.headers.get("Result", "*")
        ok = True
        ply = 0
        for san, comment in g.moves:
            fen_before = board.to_fen()
            stm = board.side
            score = parse_score(comment) if comment else None
            try:
                board.apply_san(san)
            except Exception as e:                   # noqa: BLE001
                ok = False
                n_fail += 1
                failures.append((n_games, san, f"{e}"))
                if stop_on_error:
                    raise
                break
            if score is not None:
                yield {
                    "fen": fen_before,
                    "score_cp": score,
                    "move": san,
                    "ply": ply,
                    "stm": stm,
                    "result_stm": result_to_stm(result, stm),
                }
                n_pos += 1
            ply += 1
        if ok:
            n_ok += 1
    iter_records.stats = {                            # type: ignore[attr-defined]
        "games": n_games, "ok": n_ok, "failed": n_fail,
        "positions": n_pos, "failures": failures[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("--jsonl")
    ap.add_argument("--plain")
    ap.add_argument("--check", type=int, default=0,
                    help="replay-only; print stats and first failures")
    args = ap.parse_args()

    jf = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    pf = open(args.plain, "w", encoding="utf-8") if args.plain else None

    for rec in iter_records(args.pgn):
        if jf:
            jf.write(json.dumps(rec) + "\n")
        if pf:
            pf.write(f"fen {rec['fen']}\n")
            pf.write(f"move {rec['move']}\n")
            pf.write(f"score {rec['score_cp']}\n")
            pf.write(f"ply {rec['ply']}\n")
            pf.write(f"result {rec['result_stm']}\n")
            pf.write("e\n")

    if jf:
        jf.close()
    if pf:
        pf.close()

    st = iter_records.stats                            # type: ignore[attr-defined]
    print(f"games parsed : {st['games']}")
    print(f"fully replayed: {st['ok']}  "
          f"({100*st['ok']/max(1,st['games']):.1f}%)")
    print(f"games w/ error: {st['failed']}")
    print(f"positions     : {st['positions']}")
    if st["failures"]:
        print("first failures (game#, move, error):")
        for gnum, mv, err in st["failures"][:args.check or 10]:
            print(f"  game {gnum}: {mv!r} -> {err}")


if __name__ == "__main__":
    main()
