# Milestone 1 — Musketeer Chess Mechanics & How We Handle the Differences vs. Classic Chess

**Project:** NNUE training for Musketeer Chess (Fairy-Stockfish)
**Author:** (rebuild)  **Engine:** Musketeer Fairy-Stockfish (private build, "Fixing Gating Check", 20 May 2024)

This document explains the rules of Musketeer Chess that matter for parsing the
game data and for training a neural-network evaluation (NNUE), and states
exactly how each difference from classic chess is handled in code.

---

## 1. What Musketeer Chess is

Musketeer Chess is classic chess played on the normal 8×8 board with **two extra
"Musketeer" pieces per side** that are *gated* (introduced) onto the board during
the game rather than starting on it.

- **Version 1.0** (the version playable online): both sides use the **same two**
  extra pieces, chosen from a basket of 10 predefined pieces
  (Leopard, Cannon, Unicorn, Dragon, Chancellor, Archbishop, Elephant, Hawk,
  Fortress, Spider).
- **Version 2.0** (already playable by the Fairy-Stockfish engine, not yet
  released online): **asymmetric armies** are allowed — the two sides may use
  *different piece types* and even a *different number* of extra pieces (an
  extreme case: one side gets a single very strong piece, the other up to 6–7
  weaker ones).

> **Design consequence:** we never assume the two sides share the same extra
> pieces, nor that there are exactly two. White's men and Black's men are stored
> and handled **separately**, driven entirely by the `VariantMen` PGN header.

For this project the training data always uses **Hawk (H)** and **Unicorn (U)**
for both sides, but the code path is general.

---

## 2. Piece letters and colour convention

Classic pieces use the standard letters **P R N B Q K**. The 20 remaining
alphabet letters are free for extra pieces. In our data:

| Letter | Piece   | Betza (from `VariantMen`) |
|--------|---------|---------------------------|
| H      | Hawk    | `ADGH` — leaps of 2 and 3 (alfil, dabbaba, tripper, threeleaper) |
| U      | Unicorn | `NC` — knight + camel |

**Colour convention (corrected):** **UPPERCASE = White, lowercase = Black.**
(An early draft had this inverted; white is upper case, e.g. `H`/`U` are White,
`h`/`u` are Black.)

---

## 3. FEN: 10 ranks, not 8

A classic FEN has 8 board ranks. Musketeer surrounds the 8 playing ranks with
**two gating rows**: **rank 0** holds White's waiting pieces, **rank 9** holds
Black's. So a Musketeer FEN has **10 rank-chunks**:

```
*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU******  w KQkq - 0 1
└ rank9 ┘└──────────────── 8 playing ranks ─────────────────┘└ rank0 ┘
 (black waiting)                                              (white waiting)
```

- Rank 9 `*u***h**`: Black **Unicorn waits on b9**, Black **Hawk waits on f9**
  (`*` = empty waiting square).
- Rank 0 `HU******`: White **Hawk waits on a0**, White **Unicorn waits on b0**.
- The remaining chunks are exactly the classic FEN fields: side to move (`w`),
  castling rights (`KQkq`), en-passant square (`-`), half-move clock, full-move
  number.

**Handling:** the parser splits the position into **10 piece rows**; rows 0 and
9 populate the per-side *waiting areas*, rows 1–8 populate the 8×8 board. The
five trailing fields are parsed exactly as in classic chess. A waiting piece has
**no legal moves** — it only waits to be gated or captured.

---

## 4. Gating — the core new mechanic

Each waiting piece sits **behind a specific home-rank piece** and is gated onto
the square that piece vacates **the first time that front piece moves**.

- In PGN this is written by appending `/<Piece>` to the move, e.g.
  **`Nc3/U`** = the b1-knight moves to c3 **and** the Unicorn is gated onto b1 in
  the same move. (The letter is the gated piece; the square is the vacated home
  square, always the front piece's origin.)
- If the front piece is **captured before it ever moves, the waiting piece is
  captured with it** — it never reaches the board (it can only reappear later via
  pawn promotion, see §6).
- Gating happens **at most once per waiting piece**, and gating is optional/
  forced by the front piece's own movement — a waiting piece cannot move on its
  own.

### King gating and the check exception
The King gates its piece the first time it moves — by a normal move **or by
castling**. **Exception:** if the King is **forced to move because it is in
check**, it **forfeits** the right to gate — *unless* the king move itself
**captures the checking piece**, in which case gating still happens. (This is the
exact rule the "Fixing Gating Check" engine build enforces.)

### Castling-gating (a 3-piece move)
If a waiting piece sits behind the **King or a Rook involved in castling**, it is
gated during the castling move — making castling a **three-piece** move (king +
rook + gated piece) instead of two.

- Notation: **`O-O/He`** / **`O-O-O/He`** — the letter after `/` is the gated
  piece; the trailing square (`e`) marks that gating occurred during castling.
- **Rule constraint we validate:** an initial setup may **not** have waiting
  pieces behind **both a king and a rook at the same time** (that would gate two
  pieces in one castling move). Both rooks may gate, but then the king may not —
  and vice-versa.

> **Handling:** gating is applied as a post-step of the front piece's move: after
> moving piece *X* from its home square, if a waiting piece was registered behind
> *X*, place it on the vacated square and clear it from the waiting area. Castling
> is treated as the king's (and rook's) move for this purpose, honouring the
> check-forfeit exception.

---

## 5. En passant, halfmove clock, move number

Identical to classic chess. A double pawn push sets the en-passant target square,
recorded in the field **after** the castling rights (`... KQkq e6 0 1`). The
parser must read it there and expose it to move generation so `fxe6` e.p. is
legal for exactly one ply.

---

## 6. Promotion — per-side promotion sets

A promoting pawn may become **any piece type that started the game *on that
side*** — plus the always-legal classic minors/majors. Because armies can be
asymmetric (v2.0), **White and Black have different promotion sets**.

Example: White plays with Dragon (D) and Chancellor (M); Black plays with Hawk
(H), Unicorn (U), Archbishop (A). Then:

- White may promote to **N, B, R, Q, M, D** — **not** H, U or A.
- Black may promote to **N, B, R, Q, H, U, A** — **not** M or D.

> **Handling:** promotion targets are computed per colour from that colour's
> `VariantMen` entry, never shared.

---

## 7. Betza notation — why and where

The extra pieces are unconventional, so their moves are defined generically with
**Betza notation** rather than hard-coded. Fairy-Stockfish already implements the
Betza parser (`src/pieces.cpp`); every piece's rule is given in the PGN header
field `VariantMen`, e.g.:

```
P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;
M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2
```

- **Atoms** are single leaps: `W`(1,0) `F`(1,1) `D`(2,0) `N`(2,1) `A`(2,2)
  `H`(3,0) `C`(3,1) `Z`(3,2) `G`(3,3); `B`/`R` are the sliding
  (rider) versions of `F`/`W`, `Q`=`R`+`B`, `K` = one-step all-8.
- **Modifiers** restrict/qualify: direction `f b l r v s`, `m` move-only,
  `c` capture-only, `i` initial-only, `n` non-jumping (lame leaper), `e`
  en-passant, `p` hopper, and range digits for riders (`B3` = bishop up to 3).

So a pawn `fmWfceFifmnD` = forward push (`fmW`), diagonal capture + e.p.
(`fceF`), initial double push (`ifmnD`); Hawk `ADGH` = leaps to distance 2 and 3;
Unicorn `NC` = knight + camel.

> **Handling:** we implement this Betza grammar directly (`src/betza.py`) so the
> same code both (a) disambiguates SAN moves when parsing PGN and (b) computes the
> geometric features Model 3/4 need (controlled-square counts, colour-boundness,
> infinite-range detection). The engine (`pieces.cpp`) remains the ground-truth
> validator.

---

## 8. Summary — every difference and how it is handled

| Classic chess | Musketeer difference | Our handling |
|---|---|---|
| 8-rank FEN | 10-rank FEN (rank 0 / rank 9 waiting rows) | split into 10 rows; rows 0/9 → per-side waiting areas |
| Fixed starting army | 2 extra pieces gated in (v1.0); asymmetric & variable count (v2.0) | armies read per-side from `VariantMen`; never assume symmetry or count |
| — | Gating on first move of the front piece (`Nc3/U`) | apply as post-step of the front piece's move |
| — | King gating, forfeited if forced to move by check (unless it captures the checker) | check-forfeit rule enforced; matches "Fixing Gating Check" build |
| 2-piece castling | 3-piece castling-gating (`O-O/He`) | gate during castling; forbid pieces behind king *and* rook together |
| Capture removes 1 piece | capturing an un-moved front piece removes its waiting piece too | linked capture in the waiting area |
| Promote to N/B/R/Q | promote to any piece that started on **that side** | per-colour promotion sets |
| Hard-coded piece moves | arbitrary pieces via Betza | `src/betza.py`, validated vs `pieces.cpp` |

This is the basis for Milestone 2 onward: the PGN → (FEN, evaluation, result)
parser replays every SAN move through exactly these rules and prints a board
diagram per move for verification (as advised), before the positions are encoded
as NNUE training features.
