"""
HalfKP / HalfKA feature set for Musketeer, keyed by canonical Betza id.

This is the classic NNUE "king-relative" sparse feature set, adapted so the
"piece type" dimension is a **canonical Betza rule id** (see betza_id), never a
letter.  A piece contributes the same feature whether it is called T or V, as
long as its rule is the same.

Per perspective (own-king and opponent-king), an active feature is the triple

    ( king_bucket , piece_square , piece_type , piece_colour_relative )

flattened to a single index:

    piece_plane = (type * 2 + rel_colour) * 64 + piece_square      # 0 .. T*2*64
    index       = king_bucket * (T * 2 * 64) + piece_plane

  * HalfKA  — every piece is emitted, including the royal king (``include_kings``).
  * HalfKP  — the king is used only as the anchor; king pieces are not emitted.

Squares are oriented to the perspective side (vertical mirror for Black) so the
network always sees its own men advancing up the board — the standard NNUE
convention.  ``king_buckets`` (<= 64) coarsens the king square so the input /
first-layer size stays trainable on CPU; 64 is the exact HalfK.

Gating phases (the client's three stages) are reported by :func:`gating_stage`:

    0  no side has finished gating   ("no gated pieces")
    1  exactly one side has finished gating
    2  both sides have finished gating

so a bucketed model can learn a separate head per phase.
"""
from __future__ import annotations

from musketeer import Board, WHITE, BLACK
import betza_id as BID

SQ = 64


def _orient_sq(f: int, r: int, persp: int) -> int:
    """Square index 0..63 from ``persp``'s view (vertical mirror for Black)."""
    return (r * 8 + f) if persp == WHITE else ((7 - r) * 8 + f)


def _king_square(board: Board, color: int):
    for (f, r), (letter, c) in board.board.items():
        if c == color:
            pc = board.pieces.get(letter)
            if pc is not None and pc.is_royal:
                return f, r
    return None


def num_features(reg, king_buckets: int = 64) -> int:
    return king_buckets * reg.num_types * 2 * SQ


def gating_stage(board: Board) -> int:
    """0/1/2 = number of sides that have finished gating (see module docstring)."""
    return int(not board.white_wait) + int(not board.black_wait)


def active_indices(board: Board, reg, persp: int,
                   king_buckets: int = 64, include_kings: bool = True) -> list[int]:
    """Active HalfKA/HalfKP feature indices for one perspective.

    Identity-independent: the piece-type component is ``reg.id_of_piece`` (a
    canonical Betza id), so the letter never enters the index.
    """
    krc = _king_square(board, persp)
    if krc is None:
        return []
    T = reg.num_types
    plane = T * 2 * SQ
    ksq = _orient_sq(krc[0], krc[1], persp)
    kb = ksq * king_buckets // 64
    base = kb * plane
    out: list[int] = []
    for (f, r), (letter, c) in board.board.items():
        pc = board.pieces.get(letter)
        if pc is None:
            continue
        if pc.is_royal and not include_kings:
            continue                      # HalfKP: king anchors only
        t = reg.id_of_piece(pc)           # canonical rule id, NOT the letter
        if t >= T:                        # rule outside a frozen registry
            continue
        psq = _orient_sq(f, r, persp)
        rel = 0 if c == persp else 1
        out.append(base + (t * 2 + rel) * SQ + psq)
    return out


def board_features(board: Board, reg, king_buckets: int = 64,
                   include_kings: bool = True) -> dict:
    """Return {"own": [...], "opp": [...], "stage": 0|1|2} for the side to move.

    "own" is the side-to-move perspective, "opp" the other — the two
    accumulators an NNUE concatenates before its output head.
    """
    stm = board.side
    return {
        "own": active_indices(board, reg, stm, king_buckets, include_kings),
        "opp": active_indices(board, reg, -stm, king_buckets, include_kings),
        "stage": gating_stage(board),
    }


def fen_features(fen: str, variant_men: str, reg, king_buckets: int = 64,
                 include_kings: bool = True) -> dict:
    return board_features(Board.from_fen(fen, variant_men), reg,
                          king_buckets, include_kings)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    men = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
           "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
    # start-ish position with two waiting pieces per side (H,U white; h,u black)
    fen = "*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** w KQkq - 0 1"
    reg = BID.registry_from_variants([men])

    ok = True

    def check(label, cond):
        global ok
        print(f"  [{'OK ' if cond else 'FAIL'}] {label}")
        ok = ok and cond

    ka = fen_features(fen, men, reg, include_kings=True)
    kp = fen_features(fen, men, reg, include_kings=False)
    check("HalfKA emits both kings that HalfKP omits",
          len(ka["own"]) == len(kp["own"]) + 2)      # own king + opponent king
    check("all indices in range", all(0 <= i < num_features(reg) for i in ka["own"]))
    check("start position gating stage == 0 (neither side gated)", ka["stage"] == 0)

    # a position where White has finished gating (no white waiting piece)
    fen1 = "*u******/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/******** w KQkq - 0 1"
    check("one-side-gated stage == 1", fen_features(fen1, men, reg)["stage"] == 1)
    fen2 = "********/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/******** w KQkq - 0 1"
    check("both-sides-gated stage == 2", fen_features(fen2, men, reg)["stage"] == 2)

    # ACID TEST: rename every custom letter, keep the rules -> identical feature set
    men2 = men.replace("H:ADGH", "X:ADGH").replace("U:NC", "Y:NC")
    fen_r = (fen.replace("u", "y").replace("h", "x")
             .replace("H", "X").replace("U", "Y"))
    reg2 = BID.registry_from_variants([men2])
    a = fen_features(fen, men, reg2)          # same frozen reg for both
    b = fen_features(fen_r, men2, reg2)
    check("renamed-twin HalfKA feature sets identical (own)",
          sorted(a["own"]) == sorted(b["own"]))
    check("renamed-twin HalfKA feature sets identical (opp)",
          sorted(a["opp"]) == sorted(b["opp"]))

    print(f"  num_features(64 buckets) = {num_features(reg):,}  "
          f"(king_buckets x {reg.num_types} types x 2 x 64)")
    print("\nHALFKA SELF-TEST ALL PASS" if ok else "\nFAILURES")
    raise SystemExit(0 if ok else 1)
