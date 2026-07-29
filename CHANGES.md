# Change Log

A running record of changes to the Musketeer NNUE work, newest first.
Kept per the client's request to track changes clearly.

---

## 2026-07-28 — Piece values from the engine + "directions" feature

### 1. Adopted the engine's authoritative piece values
Replaced the earlier *estimated* piece values in the feature encoder
(`src/features.py`, `PIECE_VALUE`) with the values taken from the Musketeer
Stockfish source (`types.h`, midgame values), normalised to Pawn = 100
(engine value / 1.71):

| Piece   | Old (estimate) | New (from engine) |
|---------|---------------:|------------------:|
| Pawn    | 100 | 100 |
| Knight  | 320 | 447 |
| Bishop  | 330 | 483 |
| Rook    | 500 | 750 |
| Queen   | 900 | 1462 |
| **Hawk**    | 700 | **899** |
| **Unicorn** | 550 | **926** |
| Archbishop | 650 | 1191 |
| Chancellor | 850 | 1316 |
| Amazon  | 1100 | 1918 |
| Elephant | 750 | 1035 |
| Cannon  | 600 | 1000 |
| Leopard | 550 | 964 |
| Spider  | 500 | 1357 |
| Fortress | 700 | 1144 |

Notable correction: **Unicorn (926) > Hawk (899)** — the opposite of the earlier
estimate; both are near-queen-strength leapers.

> **Note on scales.** The engine's `types.h` numbers are the **SPSA internal**
> metric (Rook ~1282, Pawn 171), not the client's **cp relative values**
> (100 = pawn) used for army selection. The internal values are legitimate for
> the network's material signal, but the "purest" material feature would use the
> client's **cp relative-value table** (1500+ piece evaluations). *Open request:
> obtain that table to use instead of / alongside the SPSA-derived values.*

### 2. Added a "number of directions" feature to Model 3
Per the client's relative-value methodology (central control, colour-bound
penalty, infinite range, **number of distinct directions a piece can move**),
added an 8th Betza-geometry feature `directions` in `src/features.py`
(`piece_geometry`). It counts distinct primitive move directions (a slider's ray
counts once): Knight = 8, Bishop = 4, Rook = 4, Queen = 8, King = 8, Hawk = 8,
Unicorn = 16 — consistent with the client's examples (knight 8, bishop 4).

Consequences:
- Model-3 encoding grew from 128 → **136 inputs**: 64 board + 8 piece-types ×
  **8** geometry features (was 7) + 8 gating-file features.
- `train/model3.py` first layer now sized from `features.N_FEATURES_M3` (136).
- No feature was dropped; the gating block is retained.

### 3. Retraining
All four models retrained on the full 835,853-position dataset with the updated
piece values (Model 3 also with the new `directions` feature). Stale feature
caches cleared; previous models backed up to `models/prev_values/`.

Final validation loss, before vs. after (lower is better):

| Model | Before | After | Change |
|-------|-------:|------:|-------:|
| Model 1 (baseline)        | 0.0336 | **0.0318** | −5.4% |
| Model 2 (gating-adaptive) | 0.0323 | **0.0310** | −4.0% |
| **Model 3 (Betza features)** | 0.0245 | **0.0238** | −2.9% (best overall) |
| Model 4 (deep)            | 0.0335 | **0.0321** | −4.2% |

Every model improved. Model 3 received both changes (engine piece values +
the `directions` feature, 136-input encoding) and retained the best evaluation,
confirming that the "number of directions" parameter from the client's
methodology adds signal to the network.

---

## Files touched
- `src/features.py` — `PIECE_VALUE` (engine values); `piece_geometry` (+`directions`);
  `encode_board_model3` (136-input layout); `N_FEATURES_M3`.
- `train/model3.py` — input layer sized from `N_FEATURES_M3`.
- `models/prev_values/` — backup of the pre-update trained models.

## Open items / requests to the client
1. The **cp relative-value table** (1500+ piece evaluations, 100 = pawn), and
   per-army value overrides, to use as the material feature.
2. Confirmation of the **feature set + network sizes** for the engine-native
   NNUE integration (see `docs/NNUE_Integration_Plan.md`).
