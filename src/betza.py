"""
Betza / XBetza notation move generator for Musketeer Chess.

This is the geometric foundation for the whole project:
  * the PGN parser uses it to disambiguate SAN moves and to know where a
    piece may go (Milestone 2);
  * the Model-3 / Model-4 feature encoders use it to compute engineered
    features such as "controlled squares at distance 1..4" and the
    colour-boundness score (Milestones 3-4).

We parse the exact `VariantMen` strings that appear in the PGN headers, e.g.

    P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;
    M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2

Coordinates
-----------
We work on the 8x8 *playing* board.  A square is a tuple ``(file, rank)`` with
``file`` and ``rank`` in ``0..7`` (a1 == (0, 0), h8 == (7, 7)).  "Forward" is
``+rank`` for White; for Black we simply negate the rank component of every
step, which is what the ``color`` argument does.

The gating rows (FEN rank 0 for White, rank 9 for Black) are *not* part of this
module -- pieces waiting to be gated have no moves.  That logic lives in
``musketeer.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# --------------------------------------------------------------------------- #
# Atom table: letter -> the (a, b) leap magnitudes.
# The full move set of an atom is every (+-a, +-b) and (+-b, +-a), deduped.
# --------------------------------------------------------------------------- #
ATOMS: dict[str, tuple[int, int]] = {
    "W": (1, 0),  # wazir      – orthogonal step
    "F": (1, 1),  # ferz       – diagonal step
    "D": (2, 0),  # dabbaba
    "N": (2, 1),  # knight
    "A": (2, 2),  # alfil
    "H": (3, 0),  # threeleaper
    "C": (3, 1),  # camel
    "Z": (3, 2),  # zebra
    "G": (3, 3),  # tripper
}

# Compound letters that expand to a rider (unlimited unless a range digit is
# given) built on a base atom.
RIDERS: dict[str, tuple[int, int]] = {
    "R": (1, 0),  # rook   – wazir-rider
    "B": (1, 1),  # bishop – ferz-rider
}
# Q and K expand to two components each.
COMPOUND: dict[str, str] = {
    "Q": "RB",  # queen  = rook + bishop
    "K": "WF",  # king   = one-step wazir + ferz (all 8, range 1)
}

DIRECTION_LETTERS = set("fblrvs")
MODIFIER_LETTERS = set("mcinep")  # move/capture/initial/nonjump/enpassant/hopper


def _leap_vectors(a: int, b: int) -> list[tuple[int, int]]:
    """All symmetric integer vectors for a leap of magnitudes (a, b)."""
    raw = {
        (sa * a, sb * b) for sa in (1, -1) for sb in (1, -1)
    } | {
        (sb * b, sa * a) for sa in (1, -1) for sb in (1, -1)
    }
    return sorted(v for v in raw if v != (0, 0))


def _classify(dx: int, dy: int) -> set[str]:
    """
    Return the set of Betza direction letters that select vector (dx, dy),
    using the Fairy-Stockfish / H.G. Muller convention (forward = +dy).

    * orthogonal (one component 0): f/b/l/r pick one direction each.
    * diagonal (|dx| == |dy|):       f/b pick the two forward/back diagonals,
                                     l/r the two left/right diagonals.
    * oblique (|dx| != |dy|, both !=0): the "narrow" component decides f/b vs
                                     l/r -- a move is forward-ish if |dy|>|dx|.
    v == f|b (vertical), s == l|r (sideways) are added for convenience.
    """
    d: set[str] = set()
    if dx == 0 or dy == 0:  # orthogonal
        if dy > 0:
            d.add("f")
        if dy < 0:
            d.add("b")
        if dx > 0:
            d.add("r")
        if dx < 0:
            d.add("l")
    elif abs(dx) == abs(dy):  # diagonal – belongs to both a vertical and a side
        d.add("f" if dy > 0 else "b")
        d.add("r" if dx > 0 else "l")
    else:  # oblique leaper (e.g. knight, camel)
        if abs(dy) > abs(dx):  # taller than wide -> vertical family
            d.add("f" if dy > 0 else "b")
        else:  # wider than tall -> sideways family
            d.add("r" if dx > 0 else "l")
    if "f" in d or "b" in d:
        d.add("v")
    if "l" in d or "r" in d:
        d.add("s")
    return d


@dataclass
class MoveComponent:
    """One parsed Betza term, e.g. ``fmW`` or ``B3`` or ``ifmnD``."""
    vectors: list[tuple[int, int]]      # base step vectors (already direction-filtered)
    rider: bool                          # True -> slide, False -> single leap
    max_range: int                       # for riders; 0 == unlimited
    can_move: bool = True                # m/c flags; default both allowed
    can_capture: bool = True
    initial_only: bool = False           # i
    non_jumping: bool = False            # n (lame leaper: midpoint must be clear)
    en_passant: bool = False             # e
    hopper: bool = False                 # p (needs a screen)


@dataclass
class Piece:
    letter: str
    components: list[MoveComponent] = field(default_factory=list)
    is_royal: bool = False
    can_castle: bool = False


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _parse_term(term: str) -> list[MoveComponent]:
    """Parse a single Betza term (modifiers + one atom + optional range)."""
    dirs: set[str] = set()
    mods: set[str] = set()
    i = 0
    while i < len(term) and (term[i] in DIRECTION_LETTERS or term[i] in MODIFIER_LETTERS):
        if term[i] in DIRECTION_LETTERS:
            dirs.add(term[i])
        else:
            mods.add(term[i])
        i += 1
    if i >= len(term):
        return []
    atom = term[i]
    i += 1
    # optional range digit(s) for riders
    range_digits = ""
    while i < len(term) and term[i].isdigit():
        range_digits += term[i]
        i += 1

    # Special: castling marker O (handled by the board, not here).
    if atom == "O":
        return []

    # Expand compound letters into multiple base atoms.
    base_letters: list[str]
    if atom in COMPOUND:
        base_letters = list(COMPOUND[atom])
    else:
        base_letters = [atom]

    comps: list[MoveComponent] = []
    for bl in base_letters:
        if bl in RIDERS:
            a, b = RIDERS[bl]
            rider = True
            default_range = 0  # unlimited
        elif bl in ATOMS:
            a, b = ATOMS[bl]
            rider = False
            default_range = 1
        elif bl in RIDERS or bl in ATOMS:
            continue
        else:
            # Unknown atom letter -> skip (keeps us robust to exotic pieces).
            continue

        max_range = int(range_digits) if range_digits else default_range
        if range_digits and not rider:
            # a ranged non-rider (rare) behaves like a rider of that atom
            rider = True

        vectors = _leap_vectors(a, b)
        if dirs:
            # Expand shorthand: v -> f,b ; s -> l,r for filtering purposes.
            wanted = set(dirs)
            if "v" in wanted:
                wanted |= {"f", "b"}
            if "s" in wanted:
                wanted |= {"l", "r"}
            vectors = [v for v in vectors if _classify(*v) & wanted]

        comps.append(
            MoveComponent(
                vectors=vectors,
                rider=rider,
                max_range=max_range,
                can_move=("c" not in mods),      # if capture-only, cannot move
                can_capture=("m" not in mods),   # if move-only, cannot capture
                initial_only=("i" in mods),
                non_jumping=("n" in mods),
                en_passant=("e" in mods),
                hopper=("p" in mods),
            )
        )
    return comps


def _split_terms(betza: str) -> list[str]:
    """Split a piece's Betza string into individual terms.

    A term is ``<modifier/direction letters><ATOM><range digits>``.  Atom
    letters are the only *uppercase* characters, modifiers/directions are
    lowercase, and digits are the (optional) rider range.  So: an uppercase
    char closes any term that already owns an atom; a lowercase char after an
    atom begins the next term; digits stick to the current atom.
    """
    terms: list[str] = []
    cur = ""
    have_atom = False
    for ch in betza:
        if ch.isupper():                 # atom letter (incl. B/R/Q/K/O)
            if have_atom:
                terms.append(cur)
                cur = ""
            cur += ch
            have_atom = True
        elif ch.isdigit():               # range digit -> stays with atom
            cur += ch
        else:                            # lowercase modifier / direction
            if have_atom:
                terms.append(cur)
                cur = ""
                have_atom = False
            cur += ch
    if cur:
        terms.append(cur)
    return terms


def parse_piece(letter: str, betza: str) -> Piece:
    """Parse one piece definition, e.g. parse_piece('H', 'ADGH')."""
    is_royal = "K" in betza and letter.upper() == "K"
    can_castle = "O" in betza
    components: list[MoveComponent] = []
    for term in _split_terms(betza):
        components.extend(_parse_term(term))
    return Piece(
        letter=letter.upper(),
        components=components,
        is_royal=is_royal,
        can_castle=can_castle,
    )


def parse_variant_men(variant_men: str) -> dict[str, Piece]:
    """Parse a full VariantMen header string into {LETTER: Piece}."""
    pieces: dict[str, Piece] = {}
    for chunk in variant_men.strip().split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        letter, betza = chunk.split(":", 1)
        pieces[letter.strip().upper()] = parse_piece(letter.strip(), betza.strip())
    return pieces


# --------------------------------------------------------------------------- #
# Move generation on an 8x8 board
# --------------------------------------------------------------------------- #
def _on_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def generate_targets(
    piece: Piece,
    file: int,
    rank: int,
    color: int,
    occupied: dict[tuple[int, int], int] | None = None,
    *,
    has_moved: bool = True,
    captures_only: bool = False,
    quiets_only: bool = False,
) -> set[tuple[int, int]]:
    """
    Pseudo-legal destination squares for ``piece`` standing on (file, rank).

    ``color``: +1 White (forward = +rank), -1 Black (forward = -rank).
    ``occupied``: map square -> color of the piece on it (+1/-1). Empty squares
        are absent.  If ``None``, the board is treated as empty (used for the
        geometric feature encoders in Model 3/4).
    ``has_moved``: gates the ``i`` (initial-only) components such as the pawn
        double push.
    """
    if occupied is None:
        occupied = {}
    targets: set[tuple[int, int]] = set()

    for comp in piece.components:
        if comp.initial_only and has_moved:
            continue
        allow_move = comp.can_move and not captures_only
        allow_cap = comp.can_capture and not quiets_only
        if not (allow_move or allow_cap):
            continue

        for (bdx, bdy) in comp.vectors:
            dx = bdx
            dy = bdy * color  # flip forward direction for Black
            steps = comp.max_range if comp.rider and comp.max_range > 0 else (
                99 if comp.rider else 1
            )
            f, r = file, rank
            blocked = False
            hopped = False  # for hoppers: have we jumped the screen yet
            for step in range(1, steps + 1):
                f, r = file + dx * step, rank + dy * step
                if not _on_board(f, r):
                    break
                occ = occupied.get((f, r))

                # Lame-leaper (non-jumping) midpoint check for a single leap.
                if comp.non_jumping and not comp.rider:
                    mid = (file + dx // 2 if dx % 2 == 0 else file + _sgn(dx),
                           rank + dy // 2 if dy % 2 == 0 else rank + _sgn(dy))
                    # only meaningful when there is a genuine intermediate square
                    if (dx % 2 == 0 and dy % 2 == 0) and mid != (f, r):
                        if occupied.get(mid) is not None:
                            break

                if comp.hopper:
                    # cannon-style: slide over empties until we meet a screen,
                    # then the next occupied square is a capture target.
                    if not hopped:
                        if occ is None:
                            continue
                        hopped = True
                        continue
                    else:
                        if occ is None:
                            continue
                        if occ != color and allow_cap:
                            targets.add((f, r))
                        break

                if occ is None:
                    if allow_move:
                        targets.add((f, r))
                    continue
                else:
                    if occ != color and allow_cap:
                        targets.add((f, r))
                    break  # blocked either way for a rider
            del blocked, hopped
    return targets


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------- #
# Tiny self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    men = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
           "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
    pieces = parse_variant_men(men)

    def show(letter, sq, color=1, **kw):
        f, r = sq
        t = generate_targets(pieces[letter], f, r, color, **kw)
        print(f"{letter}@{chr(97+f)}{r+1}: {len(t)} -> "
              f"{sorted(chr(97+tf)+str(tr+1) for tf, tr in t)}")

    print("Parsed pieces:", ", ".join(sorted(pieces)))
    # Knight on d4 (open board) should reach 8 squares.
    show("N", (3, 3))
    # Hawk H=ADGH on e4: leaps of 2 (A,D) and 3 (G,H) only.
    show("H", (4, 3))
    # Unicorn U=NC on e4: knight + camel.
    show("U", (4, 3))
    # White pawn on e2 (not moved) -> e3, e4.
    show("P", (4, 1), has_moved=False)
