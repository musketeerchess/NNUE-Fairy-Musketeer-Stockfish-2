"""
Musketeer Chess board: 10-rank FEN, SAN application, and gating.

Scope: this module *applies* the moves given in a PGN (in SAN) and reproduces
the resulting position/FEN.  It is not a full playing engine -- it does not need
to enumerate every legal move to choose one, only to (a) resolve SAN
disambiguation and (b) reject moves that would leave the mover's king in check.
Move geometry comes from ``betza.py``; the private engine is the ground-truth
validator (see ``scripts/validate_parser.py``).

Coordinates
-----------
Internal board squares are ``(file, rank)`` with ``file`` 0..7 (a..h) and
``rank`` 0..7 where rank 0 is White's back rank ("1") and rank 7 is Black's
("8").  White forward is +rank.

FEN
---
A Musketeer FEN has 10 '/'-separated rows, top (Black side) to bottom:

    row 0            -> Black waiting area  (chess "rank 9")
    rows 1..8        -> board, row 1 = chess rank 8 = internal rank 7, ...
                                 row 8 = chess rank 1 = internal rank 0
    row 9            -> White waiting area  (chess "rank 0")

followed by: side-to-move, castling rights, en-passant square, halfmove clock,
fullmove number -- exactly as in classic chess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from betza import Piece, parse_variant_men, generate_targets

WHITE, BLACK = 1, -1
CLASSIC = set("PNBRQK")


def sq_name(f: int, r: int) -> str:
    return f"{chr(97 + f)}{r + 1}"


def name_sq(s: str) -> tuple[int, int]:
    return ord(s[0]) - 97, int(s[1]) - 1


@dataclass
class Board:
    pieces: dict[str, Piece]                       # LETTER -> Piece (from VariantMen)
    board: dict[tuple[int, int], tuple[str, int]]  # (f,r) -> (LETTER, color)
    white_wait: dict[int, str]                     # file -> LETTER (waiting on rank 0)
    black_wait: dict[int, str]                     # file -> LETTER (waiting on rank 9)
    side: int = WHITE
    castling: str = "KQkq"
    ep: tuple[int, int] | None = None
    halfmove: int = 0
    fullmove: int = 1
    # Per-side promotion sets, derived from which pieces exist for each colour.
    promo_white: set[str] = field(default_factory=set)
    promo_black: set[str] = field(default_factory=set)

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_fen(cls, fen: str, variant_men: str) -> "Board":
        pieces = parse_variant_men(variant_men)
        parts = fen.split()
        rows = parts[0].split("/")
        if len(rows) != 10:
            raise ValueError(f"expected 10 FEN rows, got {len(rows)}: {parts[0]!r}")

        board: dict[tuple[int, int], tuple[str, int]] = {}
        white_wait: dict[int, str] = {}
        black_wait: dict[int, str] = {}

        def parse_wait(row: str, target: dict[int, str]) -> None:
            f = 0
            for ch in row:
                if ch == "*":
                    f += 1
                elif ch.isdigit():
                    f += int(ch)
                else:
                    target[f] = ch.upper()
                    f += 1

        parse_wait(rows[0], black_wait)   # chess rank 9
        parse_wait(rows[9], white_wait)   # chess rank 0

        for i, row in enumerate(rows[1:9]):            # rows 1..8
            rank = 7 - i                               # row1 -> internal rank 7
            f = 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                else:
                    color = WHITE if ch.isupper() else BLACK
                    board[(f, rank)] = (ch.upper(), color)
                    f += 1

        side = WHITE if parts[1] == "w" else BLACK
        castling = parts[2] if len(parts) > 2 else "-"
        ep = None if len(parts) < 4 or parts[3] == "-" else name_sq(parts[3])
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1

        # Promotion sets: any piece letter that this colour actually fields
        # (on board or waiting) plus the always-legal classic promotions.
        base = {"N", "B", "R", "Q"}
        pw, pb = set(base), set(base)
        for (letter, color) in board.values():
            (pw if color == WHITE else pb).add(letter)
        for letter in white_wait.values():
            pw.add(letter)
        for letter in black_wait.values():
            pb.add(letter)
        pw.discard("K"); pb.discard("K"); pw.discard("P"); pb.discard("P")

        return cls(pieces, board, white_wait, black_wait, side, castling, ep,
                   halfmove, fullmove, pw, pb)

    # ---- helpers --------------------------------------------------------- #
    def _occ(self) -> dict[tuple[int, int], int]:
        return {sq: col for sq, (_, col) in self.board.items()}

    def king_sq(self, color: int) -> tuple[int, int] | None:
        for sq, (letter, col) in self.board.items():
            if col == color and self.pieces.get(letter, Piece(letter)).is_royal:
                return sq
        # fall back to literal 'K'
        for sq, (letter, col) in self.board.items():
            if col == color and letter == "K":
                return sq
        return None

    def _attacked_by(self, target: tuple[int, int], attacker_color: int,
                     occ: dict[tuple[int, int], int]) -> bool:
        """Is ``target`` hit by any capture of a piece of ``attacker_color``?"""
        for sq, (letter, col) in self.board.items():
            if col != attacker_color:
                continue
            pc = self.pieces.get(letter)
            if pc is None:
                continue
            ts = generate_targets(pc, sq[0], sq[1], col, occ,
                                  has_moved=True, captures_only=True)
            if target in ts:
                return True
        return False

    def in_check(self, color: int) -> bool:
        ks = self.king_sq(color)
        if ks is None:
            return False
        return self._attacked_by(ks, -color, self._occ())

    def home_rank(self, color: int) -> int:
        return 0 if color == WHITE else 7

    def _has_moved(self, f: int, r: int, letter: str, color: int) -> bool:
        """Pawns need the initial-move gate; approximate 'not moved' by being
        on its home rank (rank 1 for White pawns, rank 6 for Black)."""
        if letter == "P":
            return r != (1 if color == WHITE else 6)
        return True

    # ---- candidate source resolution ------------------------------------ #
    def _candidates(self, letter: str, color: int, dest: tuple[int, int],
                    is_capture: bool) -> list[tuple[int, int]]:
        occ = self._occ()
        out: list[tuple[int, int]] = []
        pc = self.pieces.get(letter)
        if pc is None:
            return out
        for sq, (l, col) in self.board.items():
            if l != letter or col != color:
                continue
            hm = self._has_moved(sq[0], sq[1], letter, color)
            ts = generate_targets(pc, sq[0], sq[1], color, occ, has_moved=hm)
            if dest in ts:
                out.append(sq)
        # En passant: pawn capturing onto the ep square (empty target).
        if letter == "P" and is_capture and self.ep == dest and dest not in occ:
            fr = dest[1] - color
            for df in (-1, 1):
                s = (dest[0] + df, fr)
                if self.board.get(s) == ("P", color):
                    out.append(s)
        return out

    def _legal_filter(self, sources: list[tuple[int, int]], letter: str,
                      color: int, dest: tuple[int, int]) -> list[tuple[int, int]]:
        legal = []
        for src in sources:
            snap = self._snapshot()
            self._raw_move(src, dest, letter, color, promo=None, ep_capture=(
                letter == "P" and self.ep == dest and dest not in self._occ()))
            if not self.in_check(color):
                legal.append(src)
            self._restore(snap)
        return legal

    # ---- low-level move application ------------------------------------- #
    def _snapshot(self):
        return (dict(self.board), dict(self.white_wait), dict(self.black_wait),
                self.ep, self.castling)

    def _restore(self, snap) -> None:
        (self.board, self.white_wait, self.black_wait, self.ep,
         self.castling) = (dict(snap[0]), dict(snap[1]), dict(snap[2]),
                           snap[3], snap[4])

    def _raw_move(self, src, dest, letter, color, promo, ep_capture) -> None:
        """Move a single piece (no gating, no castling, no state bookkeeping
        beyond capture/ep/promotion). Used for legality probing and as the core
        of apply_san."""
        # gating-linked capture: capturing an un-moved home-rank front piece
        # also removes the piece waiting behind it.
        if dest in self.board:
            self._maybe_capture_waiting(dest)
        if ep_capture:
            cap_sq = (dest[0], src[1])
            self.board.pop(cap_sq, None)
        self.board.pop(src, None)
        placed = promo if promo else letter
        self.board[dest] = (placed, color)

    def _maybe_capture_waiting(self, sq: tuple[int, int]) -> None:
        f, r = sq
        occ = self.board.get(sq)
        if occ is None:
            return
        _, col = occ
        if col == WHITE and r == 0 and f in self.white_wait:
            self.white_wait.pop(f, None)
        elif col == BLACK and r == 7 and f in self.black_wait:
            self.black_wait.pop(f, None)

    def _apply_gating(self, gated: str, square: tuple[int, int], color: int) -> None:
        wait = self.white_wait if color == WHITE else self.black_wait
        # gating square is the vacated home-rank square; its file identifies
        # the waiting piece.
        f = square[0]
        if wait.get(f) == gated:
            wait.pop(f, None)
        else:  # fall back: find by letter (castling gates behind king OR rook)
            for wf, wl in list(wait.items()):
                if wl == gated:
                    wait.pop(wf, None)
                    break
        self.board[square] = (gated, color)

    # ---- SAN ------------------------------------------------------------- #
    def apply_san(self, san: str) -> None:
        color = self.side
        raw = san.strip().rstrip("+#!?")

        # split off gating suffix "/X" (or "/Xe" for castling)
        gated = None
        if "/" in raw:
            raw, gate = raw.split("/", 1)
            gated = gate[0].upper()

        prev_ep = self.ep
        self.ep = None
        capture_happened = False

        # ---- castling ---- #
        if raw in ("O-O", "0-0", "O-O-O", "0-0-0"):
            self._apply_castle(raw, color, gated)
            self._finish(color, pawn_or_capture=False)
            return

        # ---- promotion ---- #
        promo = None
        if "=" in raw:
            raw, p = raw.split("=", 1)
            promo = p[0].upper()

        # ---- piece type ---- #
        if raw[0].isupper():           # N,B,R,Q,K or extra pieces H,U,...
            letter = raw[0].upper()
            body = raw[1:]
        else:
            letter = "P"
            body = raw

        is_capture = "x" in body
        body = body.replace("x", "")
        dest = name_sq(body[-2:])
        hint = body[:-2]               # disambiguation (file and/or rank)
        hint_file = next((ord(c) - 97 for c in hint if c.islower()), None)
        hint_rank = next((int(c) - 1 for c in hint if c.isdigit()), None)

        # restore ep for candidate detection (ep capture uses prev_ep)
        self.ep = prev_ep
        cands = self._candidates(letter, color, dest, is_capture)
        if hint_file is not None:
            cands = [s for s in cands if s[0] == hint_file]
        if hint_rank is not None:
            cands = [s for s in cands if s[1] == hint_rank]
        legal = self._legal_filter(cands, letter, color, dest)
        pool = legal or cands
        if not pool:
            raise ValueError(f"no source for SAN {san!r} in {self.to_fen()}")
        src = pool[0]

        ep_capture = (letter == "P" and prev_ep == dest and dest not in self._occ())
        capture_happened = is_capture or (dest in self.board) or ep_capture

        self.ep = None
        self._raw_move(src, dest, letter, color, promo, ep_capture)

        # set new ep square on a double pawn push
        if letter == "P" and abs(dest[1] - src[1]) == 2:
            self.ep = (src[0], (src[1] + dest[1]) // 2)

        # gating: front piece vacated its home square -> drop waiting piece
        if gated is not None:
            self._apply_gating(gated, src, color)

        self._update_castling_rights(letter, color, src)
        self._finish(color, pawn_or_capture=(letter == "P" or capture_happened))

    def _apply_castle(self, raw: str, color: int, gated: str | None) -> None:
        r = self.home_rank(color)
        king_from = (4, r)
        if raw in ("O-O", "0-0"):
            king_to, rook_from, rook_to = (6, r), (7, r), (5, r)
        else:
            king_to, rook_from, rook_to = (2, r), (0, r), (3, r)
        self.board.pop(king_from, None)
        self.board.pop(rook_from, None)
        self.board[king_to] = ("K", color)
        self.board[rook_to] = ("R", color)
        if gated is not None:
            # gate onto whichever vacated square (king's or rook's) the waiting
            # piece sat behind, identified by its file.
            wait = self.white_wait if color == WHITE else self.black_wait
            for wf, wl in list(wait.items()):
                if wl == gated and wf in (king_from[0], rook_from[0]):
                    self._apply_gating(gated, (wf, r), color)
                    break
            else:
                for wf, wl in list(wait.items()):
                    if wl == gated:
                        self._apply_gating(gated, (wf, r), color)
                        break
        # castling removes both rights for this side
        self.castling = self.castling.replace(
            "K" if color == WHITE else "k", "").replace(
            "Q" if color == WHITE else "q", "")

    def _update_castling_rights(self, letter, color, src) -> None:
        if letter == "K":
            self.castling = self.castling.replace(
                "K" if color == WHITE else "k", "").replace(
                "Q" if color == WHITE else "q", "")
        elif letter == "R":
            r = self.home_rank(color)
            if src == (0, r):
                self.castling = self.castling.replace("Q" if color == WHITE else "q", "")
            elif src == (7, r):
                self.castling = self.castling.replace("K" if color == WHITE else "k", "")

    def _finish(self, color, pawn_or_capture: bool) -> None:
        self.halfmove = 0 if pawn_or_capture else self.halfmove + 1
        if color == BLACK:
            self.fullmove += 1
        self.side = -color

    # ---- output ---------------------------------------------------------- #
    def to_fen(self) -> str:
        def wait_row(wait: dict[int, str], color: int) -> str:
            cells = []
            for f in range(8):
                if f in wait:
                    l = wait[f]
                    cells.append(l if color == WHITE else l.lower())
                else:
                    cells.append("*")
            return "".join(cells)

        rows = [wait_row(self.black_wait, BLACK)]
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for f in range(8):
                cell = self.board.get((f, rank))
                if cell is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty); empty = 0
                    letter, col = cell
                    row += letter if col == WHITE else letter.lower()
            if empty:
                row += str(empty)
            rows.append(row if row else "8")
        rows.append(wait_row(self.white_wait, WHITE))

        board_str = "/".join(rows)
        side = "w" if self.side == WHITE else "b"
        cast = self.castling if self.castling else "-"
        ep = sq_name(*self.ep) if self.ep else "-"
        return f"{board_str} {side} {cast} {ep} {self.halfmove} {self.fullmove}"

    def ascii(self) -> str:
        lines = []
        for rank in range(7, -1, -1):
            row = []
            for f in range(8):
                cell = self.board.get((f, rank))
                if cell is None:
                    row.append(".")
                else:
                    letter, col = cell
                    row.append(letter if col == WHITE else letter.lower())
            lines.append(" ".join(row))
        return "\n".join(lines)


if __name__ == "__main__":
    men = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
           "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
    fen = "*u***h**/rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR/HU****** w KQkq b8 0 1"
    b = Board.from_fen(fen, men)
    print(b.ascii())
    print("FEN roundtrip:", b.to_fen())
    for mv in ["exd5", "e6", "dxe6", "Bxe6", "Nc3/U", "Nc6/U"]:
        b.apply_san(mv)
        print(f"after {mv}: {b.to_fen()}")
