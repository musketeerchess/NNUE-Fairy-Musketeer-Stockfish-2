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

from betza import Piece, parse_piece, generate_targets
from musketeer import Board, WHITE, BLACK
import betza_id as BID

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

# --------------------------------------------------------------------------- #
# Client cp piece values (Recap sheet), keyed by Betza. These are the relative
# piece values in centipawns (pawn ~ 110) and are the preferred material signal.
# --------------------------------------------------------------------------- #
import json as _json
import os as _os

_CP_PATH = _os.path.join(_os.path.dirname(__file__), "..", "data",
                         "piece_values_cp.json")
try:
    _CP_TABLE = _json.load(open(_CP_PATH, encoding="utf-8"))
except Exception:
    _CP_TABLE = {}

# ACTIVE_VALUES maps a piece LETTER to its material value for the current variant.
# Defaults to the static table; configure_piece_values() overrides it with the
# client's cp values by mapping each letter's Betza (from VariantMen) to a cp.
ACTIVE_VALUES: dict[str, float] = dict(PIECE_VALUE)


def piece_cp(betza: str):
    """Client cp (midgame) value for a Betza string, or None if not in the table."""
    v = _CP_TABLE.get(betza)
    return v["cp_mg"] if v else None


def configure_piece_values(variant_men: str) -> dict:
    """Build the letter -> cp value map for a variant from the client's table.
    Falls back to the static PIECE_VALUE for anything not found."""
    global ACTIVE_VALUES
    vm = {}
    for chunk in variant_men.split(";"):
        if ":" in chunk:
            letter, betza = chunk.split(":", 1)
            vm[letter.strip().upper()] = betza.strip()
    values = dict(PIECE_VALUE)
    for letter, betza in vm.items():
        cp = piece_cp(betza)
        if cp is not None:
            values[letter] = cp
    values["K"] = 0
    ACTIVE_VALUES = values
    return values


def _val(letter: str) -> float:
    return ACTIVE_VALUES.get(letter, PIECE_VALUE.get(letter, 500))


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
        x[idx] = sign * _val(letter) / SCALE

    # --- gating plane (64..127) ---
    # White waiting pieces gate onto (file, 0); Black's onto (file, 7).
    for f, letter in board.white_wait.items():
        of, orr = _orient(f, 0, stm)
        idx = 64 + orr * 8 + of
        sign = 1.0 if WHITE == stm else -1.0
        x[idx] = sign * _val(letter) / SCALE
    for f, letter in board.black_wait.items():
        of, orr = _orient(f, 7, stm)
        idx = 64 + orr * 8 + of
        sign = 1.0 if BLACK == stm else -1.0
        x[idx] = sign * _val(letter) / SCALE

    return x


def encode_fen(fen: str, variant_men: str) -> np.ndarray:
    return encode_board(Board.from_fen(fen, variant_men))


def gating_finished(board: Board) -> bool:
    """True once neither side has a piece left waiting to gate (Model 2 uses
    this to switch its input width 128 -> 64)."""
    return not board.white_wait and not board.black_wait


# --------------------------------------------------------------------------- #
# Betza-derived per-piece geometry — IDENTITY-INDEPENDENT (Model 3)
# --------------------------------------------------------------------------- #
# Geometry is a pure function of the *rule*, never the letter.  It is computed on
# an EMPTY board from e4 = (4, 3) by walking the parsed move components directly
# (not the occupancy simulator), so that move-only and capture-only reachability
# can be measured separately and no special rule is silently dropped.
#
# The cache is keyed on the rule's canonical signature (see betza_id), which is
# what fixes the old cross-variant bug: the same LETTER can mean different rules
# in different variants (e.g. T:AF vs T:BW vs T:vCnD), and a letter-keyed cache
# would hand one variant another variant's geometry.
#
# Feature set (identity-independent; the client's original 8 plus the special-
# rule signals he asked us not to lose):
#   d1..d4              controlled squares at Chebyshev distance 1..4
#   total               total controlled squares (move OR capture)
#   colour_boundness    fraction of controlled squares on e4's colour
#   infinite            has an unlimited rider (queen-like)
#   directions          distinct primitive move directions (knight=8, bishop=4)
#   move_total          squares reachable by a QUIET move  (0 => capture-only)
#   capture_total       squares reachable by a CAPTURE     (0 => move-only)
#   is_hopper           needs a screen to move (cannon-like)
#   is_lame             non-jumping / lame leaper (blockable midpoint)
#   forward_ratio       fraction of controlled squares that are forward (dy>0)
E4 = (4, 3)
GEOM_KEYS = ("d1", "d2", "d3", "d4", "total", "colour_boundness",
             "infinite", "directions")                       # legacy 8 (kept)
GEO_KEYS = GEOM_KEYS + ("move_total", "capture_total",
                        "is_hopper", "is_lame", "forward_ratio")   # 13 total

# cache is keyed by canonical signature (identity-independent), not by letter.
_GEOM_CACHE: dict[tuple, dict[str, float]] = {}


def _component_squares(comp) -> set[tuple[int, int]]:
    """Squares a single component reaches from e4 on an empty board.  Hoppers
    need a screen, so they contribute no squares here (flagged separately)."""
    out: set[tuple[int, int]] = set()
    if comp.hopper:
        return out
    steps = comp.max_range if (comp.rider and comp.max_range > 0) else (
        7 if comp.rider else 1)
    for (dx, dy) in comp.vectors:
        for s in range(1, steps + 1):
            f, r = E4[0] + dx * s, E4[1] + dy * s
            if not (0 <= f < 8 and 0 <= r < 8):
                break
            out.add((f, r))
    return out


def geometry_of_components(components) -> dict[str, float]:
    """Compute the identity-independent geometry dict from parsed components."""
    move_sqs: set[tuple[int, int]] = set()
    cap_sqs: set[tuple[int, int]] = set()
    infinite = hopper = lame = False
    for c in components:
        sqs = _component_squares(c)
        if c.can_move:
            move_sqs |= sqs
        if c.can_capture:
            cap_sqs |= sqs
        if c.rider and c.max_range == 0:
            infinite = True
        if c.hopper:
            hopper = True
        if c.non_jumping:
            lame = True
    allsq = move_sqs | cap_sqs
    total = len(allsq)
    dist = collections.Counter(max(abs(tf - E4[0]), abs(tr - E4[1]))
                               for tf, tr in allsq)
    e4_parity = (E4[0] + E4[1]) % 2
    bound = sum(1 for tf, tr in allsq if (tf + tr) % 2 == e4_parity)
    prim = set()
    for tf, tr in allsq:
        dx, dy = tf - E4[0], tr - E4[1]
        gg = math.gcd(abs(dx), abs(dy)) or 1
        prim.add((dx // gg, dy // gg))
    fwd = sum(1 for tf, tr in allsq if tr > E4[1])
    return {
        "d1": dist.get(1, 0), "d2": dist.get(2, 0),
        "d3": dist.get(3, 0), "d4": dist.get(4, 0),
        "total": total,
        "colour_boundness": (bound / total) if total else 0.0,
        "infinite": 1.0 if infinite else 0.0,
        "directions": len(prim),
        "move_total": len(move_sqs),
        "capture_total": len(cap_sqs),
        "is_hopper": 1.0 if hopper else 0.0,
        "is_lame": 1.0 if lame else 0.0,
        "forward_ratio": (fwd / total) if total else 0.0,
    }


def rule_geometry(betza: str) -> dict[str, float]:
    """Identity-independent geometry for a Betza rule (cached by fingerprint)."""
    sig = BID.canonical_signature(betza)
    g = _GEOM_CACHE.get(sig)
    if g is None:
        g = geometry_of_components(parse_piece("?", betza).components)
        _GEOM_CACHE[sig] = g
    return g


def piece_geometry(piece: Piece) -> dict[str, float]:
    """Geometry for an already-parsed Piece; cached by the piece's RULE
    signature so two letters with the same rule share one entry and a reused
    letter with a different rule never collides."""
    sig = BID.signature_of_piece(piece)
    g = _GEOM_CACHE.get(sig)
    if g is None:
        g = geometry_of_components(piece.components)
        _GEOM_CACHE[sig] = g
    return g


# --------------------------------------------------------------------------- #
# Identity-independent Model-3 geometry encoder (canonical-id slots).
#
# The original Model 3 laid its geometry block out with one slot per LETTER
# (MODEL3_TYPES).  That ties the input to the alphabet, so T:DW and V:DW would
# land in different slots.  This version indexes the block by the rule's
# CANONICAL ID from the frozen registry: same rule -> same slot, always, and the
# letter is used only to look the rule up.  Width is derived from the registry.
# --------------------------------------------------------------------------- #
N_GEO = len(GEO_KEYS)                                     # 13 geometry features
GEO_SCALE = {"d1": 8.0, "d2": 8.0, "d3": 8.0, "d4": 8.0, "total": 27.0,
             "colour_boundness": 1.0, "infinite": 1.0, "directions": 16.0,
             "move_total": 27.0, "capture_total": 27.0,
             "is_hopper": 1.0, "is_lame": 1.0, "forward_ratio": 1.0}

_GEO_REG = None            # frozen BetzaRegistry used by the identity-free encoders


def set_geo_registry(reg) -> None:
    """Install the frozen registry the identity-independent encoders index by."""
    global _GEO_REG
    _GEO_REG = reg


def n_features_geo(reg=None) -> int:
    reg = reg or _GEO_REG
    return 64 + reg.num_types * N_GEO + 16       # board + per-rule geometry + gating


def encode_board_geo(board: Board, betza_map: dict, reg) -> np.ndarray:
    """Model-3, identity-independent: board material (64) + per-canonical-rule
    geometry x signed count (num_types x 13) + gating cp per file (16)."""
    ntypes = reg.num_types
    x = np.zeros(64 + ntypes * N_GEO + 16, dtype=np.float32)
    stm = board.side
    gate_base = 64 + ntypes * N_GEO

    for (f, r), (letter, color) in board.board.items():
        of, orr = _orient(f, r, stm)
        sign = 1.0 if color == stm else -1.0
        x[orr * 8 + of] = sign * _val(letter) / SCALE
        betza = betza_map.get(letter, "")
        rid = reg.id_of(betza)                    # rule id, NOT the letter
        if rid >= ntypes:                         # unseen rule (registry not frozen on it)
            continue
        g = rule_geometry(betza)
        base = 64 + rid * N_GEO
        for ki, k in enumerate(GEO_KEYS):
            x[base + ki] += sign * (g[k] / GEO_SCALE[k])

    for f, letter in board.white_wait.items():
        x[gate_base + f] += (1.0 if WHITE == stm else -1.0) * _val(letter) / SCALE
    for f, letter in board.black_wait.items():
        x[gate_base + 8 + f] += (1.0 if BLACK == stm else -1.0) * _val(letter) / SCALE
    return x


def encode_fen_geo(fen: str, variant_men: str) -> np.ndarray:
    return encode_board_geo(Board.from_fen(fen, variant_men),
                            _betza_map(variant_men), _GEO_REG)


# --------------------------------------------------------------------------- #
# "Ultra" dense features: gating + Betza geometry only (no raw material plane).
# Used as the dense branch of the Ultra model, which pairs it with HalfKA.
# --------------------------------------------------------------------------- #
def n_features_gatebetza(reg=None) -> int:
    reg = reg or _GEO_REG
    return reg.num_types * N_GEO + 16          # per-rule geometry + gating (16)


def encode_board_gatebetza(board: Board, betza_map: dict, reg) -> np.ndarray:
    ntypes = reg.num_types
    x = np.zeros(ntypes * N_GEO + 16, dtype=np.float32)
    stm = board.side
    gate_base = ntypes * N_GEO
    for (f, r), (letter, color) in board.board.items():
        sign = 1.0 if color == stm else -1.0
        betza = betza_map.get(letter, "")
        rid = reg.id_of(betza)
        if rid >= ntypes:
            continue
        g = rule_geometry(betza)
        base = rid * N_GEO
        for ki, k in enumerate(GEO_KEYS):
            x[base + ki] += sign * (g[k] / GEO_SCALE[k])
    for f, letter in board.white_wait.items():
        x[gate_base + f] += (1.0 if WHITE == stm else -1.0) * _val(letter) / SCALE
    for f, letter in board.black_wait.items():
        x[gate_base + 8 + f] += (1.0 if BLACK == stm else -1.0) * _val(letter) / SCALE
    return x


def encode_fen_gatebetza(fen: str, variant_men: str) -> np.ndarray:
    return encode_board_gatebetza(Board.from_fen(fen, variant_men),
                                  _betza_map(variant_men), _GEO_REG)


# ---- legacy letter-slot Model 3 (kept for backward compatibility) ---------- #
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
        x[idx] = sign * _val(letter) / SCALE
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


# --------------------------------------------------------------------------- #
# 512-input encoding (the client's proposed NNUE layer-1 layout)
#   [  0: 64]  material: signed cp value per square, side-to-move oriented   (64)
#   [ 64: 80]  gating: white waiting cp per file (8) then black (8)          (16)
#   [ 80:512]  geometry: 24 piece types x 18 features, weighted by signed
#              on-board count. Per type: elemental atoms (W,F,D,N,A,C,Z,G,H),
#              sliders (B,R,Q), controlled squares D1..D4, colour-boundness,
#              and number of directions.                                    (432)
# Total 512. Configurable so 256 / 384 / 512 ablations are easy.
# --------------------------------------------------------------------------- #
GEOM512_TYPES = "PNBRQKHUECAFMSDLGIJOTVWZ"      # 24 piece letters
ATOMS512 = "WFDNACZGH"                          # elemental leaper atoms (9)
SLIDERS512 = "BRQ"                              # riders (3)
PER_TYPE512 = len(ATOMS512) + len(SLIDERS512) + 4 + 2      # 18
N_FEATURES_512 = 64 + 16 + len(GEOM512_TYPES) * PER_TYPE512  # 64 + 16 + 432 = 512


def _betza_map(variant_men: str) -> dict:
    m = {}
    for ch in variant_men.split(";"):
        if ":" in ch:
            letter, betza = ch.split(":", 1)
            m[letter.strip().upper()] = betza.strip()
    return m


def encode_board_512(board: Board, betza_map: dict) -> np.ndarray:
    x = np.zeros(N_FEATURES_512, dtype=np.float32)
    stm = board.side
    counts: dict[str, float] = collections.defaultdict(float)

    # material plane
    for (f, r), (letter, color) in board.board.items():
        of, orr = _orient(f, r, stm)
        sign = 1.0 if color == stm else -1.0
        x[orr * 8 + of] = sign * _val(letter) / SCALE
        counts[letter] += sign

    # gating (16): white waiting per file, then black waiting per file
    for f, letter in board.white_wait.items():
        x[64 + f] += (1.0 if WHITE == stm else -1.0) * _val(letter) / SCALE
    for f, letter in board.black_wait.items():
        x[72 + f] += (1.0 if BLACK == stm else -1.0) * _val(letter) / SCALE

    # geometry block
    base0 = 80
    for ti, letter in enumerate(GEOM512_TYPES):
        pc = board.pieces.get(letter)
        net = counts.get(letter, 0.0)
        if pc is None or net == 0.0:
            continue
        betza = betza_map.get(letter, "")
        g = piece_geometry(pc)
        desc = [1.0 if a in betza else 0.0 for a in ATOMS512]
        desc += [1.0 if s in betza else 0.0 for s in SLIDERS512]
        desc += [g["d1"] / 8, g["d2"] / 8, g["d3"] / 8, g["d4"] / 8]
        desc += [g["colour_boundness"], g["directions"] / 32.0]
        base = base0 + ti * PER_TYPE512
        for k, dv in enumerate(desc):
            x[base + k] = dv * net
    return x


def encode_fen_512(fen: str, variant_men: str) -> np.ndarray:
    return encode_board_512(Board.from_fen(fen, variant_men), _betza_map(variant_men))


if __name__ == "__main__":
    men = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
           "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
    fen = "*u***h**/rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR/HU****** w KQkq b8 0 1"
    x = encode_fen(fen, men)
    print("shape:", x.shape, "nonzero:", int((x != 0).sum()))
    print("board-plane nonzero:", int((x[:64] != 0).sum()),
          " gating-plane nonzero:", int((x[64:] != 0).sum()))
