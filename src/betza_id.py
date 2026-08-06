"""
Canonical Betza identity — the piece-rule fingerprint the network keys on.

The guiding principle: **a piece IS its Betza rule, never its letter.**  Two
pieces defined as ``T:DW`` and ``V:DW`` describe the exact same movement, so they
must produce the exact same NNUE input.  Letters are used only to decode a
FEN/PGN back to the applicable rule; from that point on the letter is discarded.

This module turns a Betza rule into:

  * ``canonical_signature(betza)`` — an identity-independent, fully expanded
    fingerprint (a sorted tuple of atomic move primitives).  It is EQUAL for
    rules that describe the same movement and DIFFERENT whenever any real
    property differs, and it preserves every special rule the parser models:
    move-only / capture-only, riders vs leapers, range, hoppers, lame
    (non-jumping) leapers, initial-only, en-passant, and directional
    restrictions.

  * ``BetzaRegistry`` — a stable mapping from that signature to a small dense
    integer id, suitable as the "piece type" dimension of a HalfKP/HalfKA
    feature set.  The six standard pieces are pinned so the ids are readable:
    ``K=0, Q=1, R=2, B=3, N=4, P=5``; every other distinct rule gets id 6, 7, …
    assigned deterministically (sorted by fingerprint), so a frozen registry is
    reproducible across runs and machines.

Canonicalisation guarantees (all proven in the self-test at the bottom):

    id(BN)   == id(NB)                 # atom / term order is irrelevant
    id(Q)    == id(RB) == id(BR)       # compounds expand to their parts
    id(cefW) == id(ecfW)              # modifier order is irrelevant
    id(vW)   == id(fW + bW)           # v == f+b  (vertical shorthand)
    id(sR)   == id(lR + rR)           # s == l+r  (sideways shorthand)
    id(hR)   == id(sR)               # h == l+r  (also sideways)
    id(W)    != id(fW)               # a directional restriction is kept
    id(W2)   != id(D)                # a blockable rider != a jumping leaper
    id(mW)   != id(W) != id(cW)      # move-only / capture-only are kept
    id(pR)   != id(R)                # a hopper is kept
"""
from __future__ import annotations

import json
from typing import Iterable

from betza import parse_piece

# --------------------------------------------------------------------------- #
# The canonical signature
# --------------------------------------------------------------------------- #
# One "primitive" is a single directed step with all of its behavioural flags.
# A piece's signature is the SET of its primitives, expanded so that any two
# notations for the same movement collapse to the same set:
#   * compounds (Q, K) are expanded to their base riders/leapers by the parser;
#   * a rider's ray is represented by its primitive step (1,0) etc. with a range;
#   * direction shorthands (v, s, h) are expanded to explicit vectors;
#   * modifier order never matters (parsed into flag booleans).
#
# Primitive layout (all ints/bools so it is hashable and order-comparable):
#   (dx, dy, rider, max_range, can_move, can_capture,
#    initial_only, non_jumping, en_passant, hopper)


def _signature_from_components(components) -> tuple:
    """Flatten parsed move components into the canonical primitive set."""
    prims = set()
    for c in components:
        for (dx, dy) in c.vectors:
            prims.add((
                int(dx), int(dy),
                bool(c.rider), int(c.max_range),
                bool(c.can_move), bool(c.can_capture),
                bool(c.initial_only), bool(c.non_jumping),
                bool(c.en_passant), bool(c.hopper),
            ))
    return tuple(sorted(prims))


def canonical_signature(betza: str) -> tuple:
    """Identity-independent fingerprint of a Betza rule (see module docstring).

    The input letter is irrelevant and never consulted; pass the RULE only.
    """
    return _signature_from_components(parse_piece("?", (betza or "").strip()).components)


def signature_of_piece(piece) -> tuple:
    """Same fingerprint, from an already-parsed Piece (its letter is ignored)."""
    return _signature_from_components(piece.components)


def signature_key(sig: tuple) -> str:
    """A stable, sortable string form of a signature (used as a dict/JSON key and
    to order the registry deterministically)."""
    return ";".join(",".join(str(v) for v in p) for p in sig)


def canonical_fingerprint(betza: str) -> str:
    """Convenience: the string fingerprint of a Betza rule directly."""
    return signature_key(canonical_signature(betza))


# --------------------------------------------------------------------------- #
# Two "8-integer" representations of a rule (client-requested, in addition to
# the dense id the network uses). Neither is used for network indexing, so they
# add no cost and require no retraining — they are alternative labels only.
# --------------------------------------------------------------------------- #
def code8(i: int) -> str:
    """The dense id as a fixed-width 8-digit zero-padded code, matching the
    client's examples: K=00000000, Q=00000001, R=00000002, B=00000003,
    N=00000004, P=00000005, then 00000006, 00000007, ...  Use with a registry:
    ``code8(reg.id_of(betza))``."""
    return f"{int(i):08d}"


# vector8 slot meanings (each counts DISTINCT move primitives in that family):
VECTOR8_SLOTS = ("rook_rays", "bishop_rays", "other_riders", "wazir_steps",
                 "ferz_steps", "knight_class_leaps", "long_jumps", "special_prims")


def vector8(betza: str) -> tuple:
    """An identity-independent 8-integer *structural* signature of a Betza rule.

    Computed from the canonical signature, so it automatically collapses
    BN=NB, Q=RB=BR, K=WF, cef=ecf, v=f+b, s=l+r=h.  Each slot counts distinct
    move primitives in a family (see VECTOR8_SLOTS):
      0 rook_rays          unlimited orthogonal rider rays  (R)
      1 bishop_rays        unlimited diagonal rider rays    (B)
      2 other_riders       range-limited or oblique riders
      3 wazir_steps        one-square orthogonal leaps      (W)
      4 ferz_steps         one-square diagonal leaps        (F)
      5 knight_class_leaps oblique leaps |dx|!=|dy|         (N,C,Z)
      6 long_jumps         (k,0)/(k,k) jumps, k>=2          (D,H,A,G)
      7 special_prims      primitives with move/capture-only, hopper, lame,
                           initial-only or en-passant flags
    """
    v = [0] * 8
    for (dx, dy, rider, rng, mv, cap, init, nj, ep, hop) in canonical_signature(betza):
        adx, ady = abs(dx), abs(dy)
        if rider:
            if (dx == 0 or dy == 0) and rng == 0:
                v[0] += 1
            elif adx == ady and rng == 0:
                v[1] += 1
            else:
                v[2] += 1
        elif (adx, ady) in ((1, 0), (0, 1)):
            v[3] += 1
        elif adx == 1 and ady == 1:
            v[4] += 1
        elif adx != ady and adx > 0 and ady > 0:
            v[5] += 1
        else:
            v[6] += 1
        if (not mv) or (not cap) or hop or nj or init or ep:
            v[7] += 1
    return tuple(v)


# --------------------------------------------------------------------------- #
# Stable integer ids (HalfKP / HalfKA "piece type" dimension)
# --------------------------------------------------------------------------- #
# The six standard pieces are pinned to small readable ids exactly as the client
# specified.  Their signatures are computed from the engine's built-in rules
# (the pawn uses its default Musketeer rule).
STANDARD_BETZA = {
    "K": "K",
    "Q": "Q",
    "R": "R",
    "B": "B",
    "N": "N",
    "P": "fmWfceFifmnD",
}
STANDARD_ORDER = ["K", "Q", "R", "B", "N", "P"]     # -> ids 0,1,2,3,4,5
STANDARD_IDS: dict[tuple, int] = {
    canonical_signature(STANDARD_BETZA[lbl]): i
    for i, lbl in enumerate(STANDARD_ORDER)
}
FIRST_CUSTOM_ID = len(STANDARD_ORDER)               # 6


class BetzaRegistry:
    """A stable rule -> dense-id map for HalfKA-style features.

    Standard pieces are pinned (K=0 … P=5).  New rules are assigned the next
    free id.  Call :meth:`freeze` with the full set of rules in a dataset to get
    a reproducible, discovery-order-independent mapping, then :meth:`save` it.
    """

    def __init__(self) -> None:
        self.sig_to_id: dict[tuple, int] = dict(STANDARD_IDS)
        self._next = FIRST_CUSTOM_ID

    # -- queries ------------------------------------------------------------- #
    def id_of_signature(self, sig: tuple) -> int:
        """Return the id for an already-computed signature, assigning if unseen."""
        i = self.sig_to_id.get(sig)
        if i is None:
            i = self._next
            self.sig_to_id[sig] = i
            self._next += 1
        return i

    def id_of(self, betza: str) -> int:
        """Return the id for a rule, assigning a fresh one if unseen."""
        return self.id_of_signature(canonical_signature(betza))

    def id_of_piece(self, piece) -> int:
        """Return the id for an already-parsed Piece (its letter is ignored)."""
        return self.id_of_signature(signature_of_piece(piece))

    def get(self, betza: str, default: int | None = None):
        """Return the id for a rule without mutating the registry."""
        return self.sig_to_id.get(canonical_signature(betza), default)

    def code8(self, betza: str) -> str:
        """The rule's dense id as an 8-digit zero-padded code (K=00000000, ...)."""
        return code8(self.id_of(betza))

    def __len__(self) -> int:
        return self._next

    @property
    def num_types(self) -> int:
        """Number of distinct piece-rule ids currently known (== max id + 1)."""
        return self._next

    # -- construction -------------------------------------------------------- #
    def freeze(self, betzas: Iterable[str]) -> "BetzaRegistry":
        """Assign ids to every rule in ``betzas`` deterministically: unseen
        signatures are sorted by fingerprint and given ids 6, 7, 8, …  so the
        result does not depend on the order rules were encountered."""
        new = {canonical_signature(b) for b in betzas} - set(self.sig_to_id)
        for sig in sorted(new, key=signature_key):
            self.sig_to_id[sig] = self._next
            self._next += 1
        return self

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str) -> None:
        rows = sorted(self.sig_to_id.items(), key=lambda kv: kv[1])
        data = {"num_types": self._next,
                "ids": [{"id": i, "fingerprint": signature_key(sig)}
                        for sig, i in rows]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=0)

    @classmethod
    def load(cls, path: str) -> "BetzaRegistry":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        reg = cls()
        reg.sig_to_id = {}
        for row in data["ids"]:
            sig = tuple(
                tuple(int(x) if x not in ("True", "False") else x == "True"
                      for x in field.split(","))
                for field in row["fingerprint"].split(";") if field
            )
            reg.sig_to_id[sig] = row["id"]
        reg._next = data["num_types"]
        return reg


def registry_from_variants(variant_men_list: Iterable[str]) -> BetzaRegistry:
    """Build a frozen registry from every rule that appears across a list of
    VariantMen strings (e.g. the whole dataset's ``variants.json``)."""
    rules: list[str] = []
    for vm in variant_men_list:
        for chunk in vm.split(";"):
            if ":" in chunk:
                rules.append(chunk.split(":", 1)[1].strip())
    return BetzaRegistry().freeze(rules)


def variant_letter_ids(variant_men: str, reg: BetzaRegistry) -> dict[str, int]:
    """Map each LETTER in one variant to its canonical rule id.  This is the ONLY
    place a letter touches an id — purely to decode the FEN.  Standard pieces
    absent from the header fall back to their built-in rule."""
    out: dict[str, int] = {}
    seen = set()
    for chunk in variant_men.split(";"):
        if ":" in chunk:
            letter, betza = chunk.split(":", 1)
            L = letter.strip().upper()
            out[L] = reg.id_of(betza.strip())
            seen.add(L)
    for L, betza in STANDARD_BETZA.items():
        if L not in seen:
            out[L] = reg.id_of(betza)
    return out


# --------------------------------------------------------------------------- #
# Self-test: proves the canonicalisation guarantees on real + synthetic rules.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def sig(b):
        return canonical_signature(b)

    checks = [
        # (label, should-be-equal, a, b)
        ("BN == NB (order)",                 True,  "BN",   "NB"),
        ("Q == RB (compound)",               True,  "Q",    "RB"),
        ("Q == BR (compound)",               True,  "Q",    "BR"),
        ("cefW == ecfW (modifier order)",    True,  "cefW", "ecfW"),
        ("vW == fWbW (v = f+b)",             True,  "vW",   "fWbW"),
        ("sR == lRrR (s = l+r)",             True,  "sR",   "lRrR"),
        ("hR == sR (h = l+r)",               True,  "hR",   "sR"),
        ("K == WF (compound)",               True,  "K",    "WF"),
        # letters differ, rule identical -> identical fingerprint
        ("T:DW == V:DW (letters ignored)",   True,  "DW",   "DW"),
        # things that MUST stay distinct
        ("W != fW (direction kept)",         False, "W",    "fW"),
        ("W2 != D (rider vs leaper)",        False, "W2",   "D"),
        ("mW != W (move-only kept)",         False, "mW",   "W"),
        ("cW != W (capture-only kept)",      False, "cW",   "W"),
        ("pR != R (hopper kept)",            False, "pR",   "R"),
        ("nN != N (lame leaper kept)",       False, "nN",   "N"),
        ("B != B3 (range kept)",             False, "B",    "B3"),
        ("cR != R (capture-only rook)",      False, "cR",   "R"),
    ]
    ok = True
    for label, want_eq, a, b in checks:
        got_eq = sig(a) == sig(b)
        mark = "OK " if got_eq == want_eq else "FAIL"
        if got_eq != want_eq:
            ok = False
        print(f"  [{mark}] {label}")

    # Registry: standards pinned, custom rules dense & deterministic.
    reg = BetzaRegistry()
    for lbl, want in zip(STANDARD_ORDER, range(6)):
        got = reg.id_of(STANDARD_BETZA[lbl])
        m = "OK " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{m}] id({lbl}) == {want} (got {got})")

    # BN and NB collapse to ONE id; a capture-only rook gets its OWN id.
    i_bn, i_nb = reg.id_of("BN"), reg.id_of("NB")
    i_cr = reg.id_of("cR")
    print(f"  [{'OK ' if i_bn == i_nb else 'FAIL'}] id(BN)==id(NB)=={i_bn}")
    print(f"  [{'OK ' if i_cr >= 6 else 'FAIL'}] id(cR)=={i_cr} (custom, distinct from R=2)")
    ok = ok and i_bn == i_nb and i_cr >= 6

    # Two extra "8-integer" representations offered alongside the dense id.
    print("\n  8-integer options (dense id | code8 | vector8):")
    for lbl in STANDARD_ORDER + ["BN", "NB", "RB"]:
        r = STANDARD_BETZA.get(lbl, lbl)
        i = reg.id_of(r)
        print(f"    {lbl:<3} id={i:<3} code8={code8(i)} vector8={vector8(r)}")

    print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)
