# Musketeer configuration for variant-nnue-pytorch.
# Copy this over third_party/variant-nnue-pytorch/variant.py to build an
# engine-native .nnue for Musketeer (Hawk + Unicorn).
#
# vs. the stock (chess) file: PIECE_TYPES 6 -> 8 (adds Hawk, Unicorn) and the
# Hawk/Unicorn values are added.  The two extra piece types are why the
# per-king feature planes grow relative to standard chess (the "128 vs 64"
# point in Milestone 1).  Piece values below are first estimates -- confirm
# with the client.

RANKS = 8
FILES = 8
SQUARES = RANKS * FILES
KING_SQUARES = RANKS * FILES

# 8 piece types: P N B R Q K + Hawk + Unicorn
PIECE_TYPES = 8
PIECES = 2 * PIECE_TYPES

# Musketeer extra pieces wait on rank 0 / rank 9 before gating.  If the chosen
# feature set represents them as pocket-style men, enable pockets:
USE_POCKETS = False
POCKETS = 2 * FILES if USE_POCKETS else 0

# Stockfish-internal value scale (pawn ~= 126).  Hawk/Unicorn estimated.
PIECE_VALUES = {
    1 : 126,    # Pawn
    2 : 781,    # Knight
    3 : 825,    # Bishop
    4 : 1276,   # Rook
    5 : 2538,   # Queen
    6 : 900,    # Hawk    (ADGH)  -- estimate, confirm
    7 : 800,    # Unicorn (NC)    -- estimate, confirm
}
