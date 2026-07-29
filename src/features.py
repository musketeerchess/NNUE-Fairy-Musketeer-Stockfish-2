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
import math

import numpy as np

from betza import Piece, generate_targets
from musketeer import Board, WHITE, BLACK

# Authoritative Musketeer piece values, taken from the Musketeer Stockfish
# engine source (`types.h`, midgame values on a Pawn=171 scale) and normalised
# here to Pawn=100 (value_engine / 1.71, rounded).  Note Unicorn (926) is
# slightly stronger than Hawk (899), and both are near-queen-in-power leapers.
PIECE_VALUE: dict[str, int] = {
    "P": 100, "N": 447, "B": 483, "R": 750, "Q": 1462, "K": 0,
    "H": 899, "U": 926,                      # Hawk, Unicorn (Musketeer)
    # remaining basket pieces (engine types.h, same normalisation)
    "E": 1035, "C": 1000, "A": 1191, "F": 1144, "M": 1316, "S": 1357,
    "D": 1918, "L": 964, "J": 900, "G": 900, "I": 900, "O": 900,
    "T": 900, "V": 900, "W": 900, "X": 900, "Y": 900, "Z": 900,
}
SCALE = 1000.0          # normalise values (Pawn -> 0.1); Amazon peaks near 1.9
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
# 8th feature "directions" = number of distinct directions the piece can move in
# (knight=8, bishop=4, ...), matching the client's relative-value methodology.
GEOM_KEYS = ("d1", "d2", "d3", "d4", "total", "colour_boundness",
             "infinite", "directions")


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
    # distinct directions: reduce every controlled vector to its primitive
    # direction (divide by gcd) so a slider's ray counts once, then count.
    prim = set()
    for tf, tr in targets:
        dx, dy = tf - E4[0], tr - E4[1]
        gg = math.gcd(abs(dx), abs(dy)) or 1
        prim.add((dx // gg, dy // gg))
    g = {
        "d1": dist.get(1, 0), "d2": dist.get(2, 0),
        "d3": dist.get(3, 0), "d4": dist.get(4, 0),
        "total": total,
        "colour_boundness": (bound / total) if total else 0.0,
        "infinite": 1.0 if infinite else 0.0,
        "directions": len(prim),
    }
    _GEOM_CACHE[piece.letter] = g
    return g


# Piece types this variant fields, in a fixed order for the feature layout.
MODEL3_TYPES = ("P", "N", "B", "R", "Q", "K", "H", "U")
_GEOM_SCALE = {"d1": 8.0, "d2": 8.0, "d3": 8.0, "d4": 8.0, "total": 27.0,
               "colour_boundness": 1.0, "infinite": 1.0, "directions": 16.0}
N_GEOM = len(GEOM_KEYS)                                   # 8 geometry features
N_FEATURES_M3 = 64 + len(MODEL3_TYPES) * N_GEOM + 8       # 64 + 64 + 8 = 136


def set_model3_types(types) -> None:
    """Override which 8 piece types the Model-3 geometry block encodes (e.g.
    for asymmetric armies with custom pieces).  Count must stay the same so the
    input dimension (N_FEATURES_M3) is unchanged."""
    global MODEL3_TYPES
    types = tuple(t.strip().upper() for t in types)
    if len(types) != len(MODEL3_TYPES):
        raise ValueError(f"need exactly {len(MODEL3_TYPES)} types, got {len(types)}")
    MODEL3_TYPES = types


def encode_board_model3(board: Board) -> np.ndarray:
    """
    Model-3 input vector (length N_FEATURES_M3 = 136):

      [  0: 64]  board material plane, signed & side-to-move oriented       (64)
      [ 64:128]  8 piece types x 8 Betza-geometry features (incl. "directions"),
                 each weighted by the signed on-board count of that type     (64)
      [128:136]  gating presence per file: own waiting +1, opp waiting -1     (8)
    """
    x = np.zeros(N_FEATURES_M3, dtype=np.float32)
    stm = board.side
    gate_base = 64 + len(MODEL3_TYPES) * N_GEOM           # = 128

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
        base = 64 + ti * N_GEOM
        net = counts.get(letter, 0)
        for ki, k in enumerate(GEOM_KEYS):
            x[base + ki] = (g[k] / _GEOM_SCALE[k]) * net

    # gating presence per file
    for f, _ in board.white_wait.items():
        x[gate_base + f] += (1.0 if WHITE == stm else -1.0)
    for f, _ in board.black_wait.items():
        x[gate_base + f] += (1.0 if BLACK == stm else -1.0)
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
