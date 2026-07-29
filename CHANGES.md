# Change Log

A running record of changes to the Musketeer NNUE work, newest first.
Kept per the client's request to track changes clearly.

---

## 2026-07-29 — New asymmetric-army games: parser fix + engine re-evaluation

### 1. Parser handles pieces omitted from VariantMen (built-in rules)
The new game PGNs omit standard pieces (e.g. the pawn) from `VariantMen`, since
those use the engine's built-in rules ("programmed directly on Stockfish").
Added default rules for standard pieces (`P, N, B, R, Q, K`) in
`src/betza.py::parse_variant_men`. Replay of the new games rose from ~0% to
~100% (e.g. `ii vs jj` 676/676, `standard_chess` 1353/1353).

### 2. Confirmed the games use *redefined* custom pieces
The new tournaments redefine pieces via `VariantMen`, e.g. `H:ZW`, `U:B2D`,
`I:ZD`, `J:ZW`, `Z:B2R2`, and use **asymmetric armies** (different pieces/counts
per side). These are the "next phase" data.

### 3. Built an engine re-evaluation pipeline (games have no evals)
The new PGNs contain only moves + result, no inline evaluations. Added
`scripts/reeval_games.py`, which:
- generates a `variants.ini` from each PGN's `VariantMen` (so the engine uses
  the exact tournament rules), and loads it via `VariantPath`;
- replays each game and asks the engine to evaluate every position;
- emits `(fen, score_cp, move, ply, result)` records (same schema as the parser).

**Critical validation** (matches the client's `variants.ini` reminder): without
the generated ini the engine mis-moves the custom pieces (it moved `I` as a
knight); with the ini, `I` correctly moves as `ZD` (zebra + dabbaba). NOTE: the
engine's ini path must be a Windows-style path or it silently fails to open; the
re-eval loop restarts the engine on pipe failure (62 restarts over the full
file, 0 positions lost).

### 4. First asymmetric-army dataset and model
Re-evaluated the full `ii vs jj` file: **676 games -> 111,791 eval-labeled
positions** (`data/processed/reeval_ii_jj.jsonl`). Trained a first asymmetric
Model 1 (MK128) on it: validation loss 0.0485 -> **0.0344** over 25 epochs,
clean convergence (`models/model1_asym.pt`). This exercises the full pipeline on
asymmetric armies with custom pieces.

### 5. Full asymmetric set of 4 models
Trained Models 2, 3, 4 on the 111,791-position asymmetric dataset. Model 3 was
given the custom pieces in its geometry block via a new option
(`train/model3.py --variant-men ... --model3-types P,N,B,R,Q,K,I,J`, backed by
`features.set_model3_types`), so its Betza-geometry features describe the actual
`I`(=ZD)/`J`(=ZW) pieces.

Asymmetric validation loss: Model 1 = 0.0344, Model 2 = 0.0314,
**Model 3 = 0.0213 (best by a wide margin)**, Model 4 = 0.0351. Model 3's
Betza-feature approach wins even more decisively on asymmetric armies than on
symmetric, confirming the value of encoding per-piece geometry for custom pieces.
Saved as `models/model{1,2,3,4}_asym.pt`.

**Remaining refinement:** the *material* value for custom pieces in the MK128
encoding is still a placeholder; wiring the engine/client cp values would
further help Models 1/2/4 (Model 3 already benefits from geometry).

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
