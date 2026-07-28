"""
Feature encoders for the Musketeer NNUE models.

Milestone-2 / Model-1 uses the **MK128** encoding described below.  The design
follows the client's specification ("128 inputs ... here the first input is 64
features"; Model 2 later "switches back to 64 inputs once gating is finished"):

    MK128  =  64 board features  +  64 gating features   (=> 128 inputs)

  * Board plane  (idx  0..63): one signed value per board square, from the
    **side-to-move's perspective** (the board is mirrored when Black is to
    move so the network always "sees" its own pieces advancing up the board).
    Value = +piece_value for our pieces, -piece_value for the opponent's,
    divided by a scale.  This is the "64 features" the client refers to.

  * Gating plane (idx 64..127): for every extra piece still WAITING (not yet
    gated), a signed value is written on the square it will gate onto (its home
    square).  +value for our waiting pieces, -value for the opponent's.  Once
    all extra pieces are gated this plane is all-zero, i.e. the position is
    effectively described by 64 features again -- exactly the "128 -> 64 after
    gating" behaviour Model 2 formalises.

The encoding is deliberately modular: swap ``encode_fen`` for a richer scheme
(e.g. the sparse HalfKA set, or the Betza-derived features Model 3 needs)
without touching the training loop.
"""

from __future__ import annotations

import collections

import numpy as np

from betza import Piece, generate_targets
from musketeer import Board, WHITE, BLACK

# Approximate Musketeer piece values (centipawns).  Exact numbers matter little
# to a learned net -- they mainly give each piece type a distinct signature.
PIECE_VALUE: dict[str, int] = {
    "P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0,
    "H": 700, "U": 550,                      # Hawk, Unicorn (Musketeer)
    # remaining basket pieces, for generality
    "E": 750, "C": 600, "A": 650, "F": 700, "M": 850, "S": 500,
    "D": 1100, "L": 550, "J": 500, "G": 500, "I": 500, "O": 500,
    "T": 500, "V": 500, "W": 500, "X": 500, "Y": 500, "Z": 500,
}
SCALE = 1000.0          # normalise values into roughly [-1.1, 1.1]
N_FEATURES = 128


def _orient(f: int, r: int, stm: int) -> tuple[int, int]:
    """Board square as seen by the side to move (mirror vertically for Black)."""
    return (f, r) if stm == WHITE else (f, 7 - r)


def encode_board(board: Board) -> np.ndarray:
    """Return the MK128 feature vector for ``board`` (float32, len 128)."""
    x = np.zeros(N_FEATURES, dtype=np.float32)
    stm = board.side

    # --- board plane (0..63) ---
    for (f, r), (letter, color) in board.board.items():
        of, orr = _orient(f, r, stm)
        idx = orr * 8 + of
        sign = 1.0 if color == stm else -1.0
        x[idx] = sign * PIECE_VALUE.get(letter, 500) / SCALE

    # --- gating plane (64..127) ---
    # White waiting pieces gate onto (file, 0); Black's onto (file, 7).
    for f, letter in board.white_wait.items():
        of, orr = _orient(f, 0, stm)
        idx = 64 + orr * 8 + of
        sign = 1.0 if WHITE == stm else -1.0
        x[idx] = sign * PIECE_VALUE.get(letter, 500) / SCALE
    for f, letter in board.black_wait.items():
        of, orr = _orient(f, 7, stm)
        idx = 64 + orr * 8 + of
        sign = 1.0 if BLACK == stm else -1.0
        x[idx] = sign * PIECE_VALUE.get(letter, 500) / SCALE

    return x


def encode_fen(fen: str, variant_men: str) -> np.ndarray:
    return encode_board(Board.from_fen(fen, variant_men))


def gating_finished(board: Board) -> bool:
    """True once neither side has a piece left waiting to gate (Model 2 uses
    this to switch its input width 128 -> 64)."""
    return not board.white_wait and not board.black_wait


# --------------------------------------------------------------------------- #
# Betza-derived per-piece geometry (Model 3, Milestone 4)
# --------------------------------------------------------------------------- #
# Computed once per piece type on an EMPTY board with the piece on e4 = (4, 3),
# exactly as the contract describes:
#   * number of controlled squares at distance 1, 2, 3, 4 (Chebyshev distance);
#   * colour-boundness = (# controlled squares of e4's colour) / (total);
#   * infinite-range flag (does the piece have an unlimited rider, like a queen).
E4 = (4, 3)
GEOM_KEYS = ("d1", "d2", "d3", "d4", "total", "colour_boundness", "infinite")


_GEOM_CACHE: dict[str, dict[str, float]] = {}


def piece_geometry(piece: Piece) -> dict[str, float]:
    cached = _GEOM_CACHE.get(piece.letter)
    if cached is not None:
        return cached
    targets = generate_targets(piece, E4[0], E4[1], WHITE, occupied=None,
                               has_moved=True)
    dist = collections.Counter(max(abs(tf - E4[0]), abs(tr - E4[1]))
                               for tf, tr in targets)
    total = len(targets)
    e4_parity = (E4[0] + E4[1]) % 2
    bound = sum(1 for tf, tr in targets if (tf + tr) % 2 == e4_parity)
    infinite = any(c.rider and c.max_range == 0 for c in piece.components)
    g = {
        "d1": dist.get(1, 0), "d2": dist.get(2, 0),
        "d3": dist.get(3, 0), "d4": dist.get(4, 0),
        "total": total,
        "colour_boundness": (bound / total) if total else 0.0,
        "infinite": 1.0 if infinite else 0.0,
    }
    _GEOM_CACHE[piece.letter] = g
    return g


# Piece types this variant fields, in a fixed order for the feature layout.
MODEL3_TYPES = ("P", "N", "B", "R", "Q", "K", "H", "U")
_GEOM_SCALE = {"d1": 8.0, "d2": 8.0, "d3": 8.0, "d4": 8.0, "total": 27.0,
               "colour_boundness": 1.0, "infinite": 1.0}


def encode_board_model3(board: Board) -> np.ndarray:
    """
    Model-3 128-input vector (always 128, regardless of gating):

      [  0: 64]  board material plane, signed & side-to-move oriented   (64)
      [ 64:120]  8 piece types x 7 Betza-geometry features, each weighted
                 by the signed on-board count of that type (own +, opp -) (56)
      [120:128]  gating presence per file: own waiting +1, opp waiting -1  (8)
    """
    x = np.zeros(128, dtype=np.float32)
    stm = board.side

    # board plane
    counts: dict[str, int] = collections.defaultdict(int)
    for (f, r), (letter, color) in board.board.items():
        of, orr = _orient(f, r, stm)
        idx = orr * 8 + of
        sign = 1.0 if color == stm else -1.0
        x[idx] = sign * PIECE_VALUE.get(letter, 500) / SCALE
        counts[letter] += sign  # net (own - opp) count per type

    # geometry x signed-count
    for ti, letter in enumerate(MODEL3_TYPES):
        pc = board.pieces.get(letter)
        if pc is None:
            continue
        g = piece_geometry(pc)
        base = 64 + ti * 7
        net = counts.get(letter, 0)
        for ki, k in enumerate(GEOM_KEYS):
            x[base + ki] = (g[k] / _GEOM_SCALE[k]) * net

    # gating presence per file
    for f, _ in board.white_wait.items():
        x[120 + f] += (1.0 if WHITE == stm else -1.0)
    for f, _ in board.black_wait.items():
        x[120 + f] += (1.0 if BLACK == stm else -1.0)
    return x


def encode_fen_model3(fen: str, variant_men: str) -> np.ndarray:
    return encode_board_model3(Board.from_fen(fen, variant_men))


if __name__ == "__main__":
    men = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
           "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
    fen = "*u***h**/rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR/HU****** w KQkq b8 0 1"
    x = encode_fen(fen, men)
    print("shape:", x.shape, "nonzero:", int((x != 0).sum()))
    print("board-plane nonzero:", int((x[:64] != 0).sum()),
          " gating-plane nonzero:", int((x[64:] != 0).sum()))
