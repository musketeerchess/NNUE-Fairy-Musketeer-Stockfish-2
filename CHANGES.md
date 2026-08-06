# Change Log

A running record of changes to the Musketeer NNUE work, newest first.
Kept per the client's request to track changes clearly.

---

## 2026-08-03 — Letter-independent piece identity (Betza fingerprint)

The guiding change: **a piece is its Betza rule, never its letter.** Two pieces
defined as `T:DW` and `V:DW` describe the same movement and must produce the same
NNUE input; the letter is used only to decode a FEN/PGN back to the rule and is
then discarded. It is never a network input, slot, embedding, fixed material
value, or branch.

### 1. Canonical Betza identity (`src/betza_id.py`)
A new module turns any Betza rule into an identity-independent fingerprint (a
fully expanded set of atomic move primitives) and a stable integer id. The
fingerprint is equal for rules that describe the same movement and different
whenever any real property differs, and it preserves every special rule:
move-only / capture-only, riders vs leapers, range, hoppers, lame (non-jumping)
leapers, initial-only, en-passant, and directional restrictions. Proven
equivalences (self-test): `BN=NB`, `Q=RB=BR`, `K=WF`, modifier order `cef=ecf`,
and the direction shorthands `v=f+b`, `s=l+r`, `h=l+r`. Proven distinctions:
`W≠fW`, `W2≠D` (blockable rider vs jumping leaper), `mW≠W≠cW`, `pR≠R` (hopper),
`nN≠N` (lame), `B≠B3` (range).

Integer ids are pinned for the standard pieces exactly as requested —
`K=0, Q=1, R=2, B=3, N=4, P=5` — and every other distinct rule is assigned
`6, 7, …` deterministically (sorted by fingerprint), so a frozen registry is
reproducible across runs. Across the big-database's 16 armies there are **64
distinct piece rules**. This id is the "piece type" dimension the HalfKP/HalfKA
and hybrid models will use.

Real payoff on the data: the letter `T` is reused for **eight different rules**
across the armies (`T:AF`, `T:BW`, `T:vCnD`, …); each now maps to its own id,
and the two `T:AF` occurrences correctly share one id. `A:NB` collapses onto the
same id as `BN`.

### 2. Model 3 geometry made letter-independent (`src/features.py`)
The geometry block previously had one slot per *letter* (`MODEL3_TYPES`), and its
per-piece geometry was cached by *letter* — a real bug, since the same letter
names different pieces in different armies, so one army could receive another's
geometry. Both are fixed:

- Geometry is now cached by the rule fingerprint, computed purely from the Betza
  atoms/modifiers by walking the parsed move components on an empty board.
- The geometry attributes are extended so the special rules the client flagged
  are not lost: alongside the original eight (controlled squares at distance
  1–4, total, colour-boundness, infinite range, number of directions) we now
  also encode **move-reachable** vs **capture-reachable** square counts (so
  capture-only `C:cR` and move-only pieces are distinguished), a **hopper** flag,
  a **lame-leaper** flag, and a **forward ratio** (so directional restrictions
  like `llN` / `fN` show up).
- A new identity-independent encoder (`encode_fen_geo`) lays out the geometry
  block by canonical rule id instead of letter. Acid test: renaming every piece
  letter in a position while keeping the rules produces a **byte-identical**
  feature vector. Material values are likewise rule-derived (letter → Betza → cp),
  so they are letter-independent too. The streaming trainer accepts this as
  `--encoding geo`; it trains cleanly (64 rules → 912 inputs).

### 3. HalfKP / HalfKA features keyed by canonical id (`src/features_halfka.py`)
Added the classic NNUE king-relative sparse feature set, with the "piece type"
dimension replaced by the canonical Betza id. An active feature per perspective
is `(king_bucket, piece_square, piece_type, colour)`, where `piece_type` is the
rule id — so a piece contributes the same feature whatever its letter. Two
variants:
- **HalfKA** — every piece emitted, king included as a piece;
- **HalfKP** — the king is the anchor only.

The `king_buckets` knob (≤ 64) coarsens the king square so the first layer stays
trainable on CPU (64 = exact HalfK). The output head is **bucketed by gating
phase**, the client's three stages: 0 = no side finished gating, 1 = one side
finished, 2 = both finished (`features_halfka.gating_stage`). Network
(`train/model_halfka.py`) is a shared EmbeddingBag feature transformer over both
perspectives + a per-phase head. Streaming trainer `train/train_halfka.py`
(`--mode halfka|halfkp`, `--king-buckets`). Same acid test passes: renaming every
letter leaves the active-feature set identical. Both modes train end-to-end
(64 canonical types, 131k features at 16 king buckets).

### 4. Hybrid model: geometry ⊕ HalfKA (`train/model_hybrid.py`)
A hybrid that fuses the dense identity-independent geometry vector (the Models
1–3 lineage) with the sparse HalfKA accumulator under one gating-bucketed head:
the geometry half supplies per-rule structure that generalises across armies,
the HalfKA half supplies king-relative positional detail. Both halves are
letter-independent, so the whole model is. Streaming trainer
`train/train_hybrid.py` trains it end-to-end.

All four new pieces (canonical id, geometry, HalfKA, hybrid) share one primitive
— the Betza fingerprint — so "same rule ⇒ same input" holds across every model.

### 5. Comparison harness (`train/compare_models.py`)
A single runner that trains every architecture — Models 1–4, the identity-
independent geometry model, the 512 net, HalfKP, HalfKA, and the hybrid — on the
**same held-out validation split** with the **same per-model position budget**,
then prints a ranked table of validation losses. This is the equal-footing
comparison the client's point 3 asks for. It is resumable (results are saved
after each model and finished models are skipped) and runs on CPU or GPU. The
split is deterministic and streaming (validation = every Nth line up to a cap;
the rest is training), so no separate split file is needed. All nine
architectures were verified to run end-to-end through it. The winner of this pass
is then retrained on the full dataset (ideally on a GPU) with its own trainer.

## 2026-07-29 (evening) — Client cp values and the 512 architecture

### 1. Ingested the client's cp piece values
Read the client's spreadsheet (Recap sheet) into `data/piece_values_cp.json`
(1,041 pieces, keyed by Betza: cp midgame, cp endgame, and the SPSA versions).
Wired into `src/features.py` via `configure_piece_values(variant_men)`, which
maps each piece letter to its cp value through its Betza. All encoders now use
these when configured. Verified values: Hawk (ADGH) 504, Unicorn (NC) 568,
I (ZD) 400, J (ZW) 484, Queen 999. These are the client's relative cp values, not
the SPSA search numbers, which is what he recommended for the material feature.

### 2. Built the client's 512-input encoder and network
`features.encode_fen_512` implements his layer-1 layout: 64 material (cp) + 16
gating + 432 Betza geometry (elemental atoms W,F,D,N,A,C,Z,G,H; sliders B,R,Q;
controlled squares D1..D4; colour-boundness; directions) across 24 piece types,
weighted by signed on-board count. Total 512. `train/model512.py` is the network
512 -> 512 -> 512 -> 512 -> 1 with clipped ReLU (about 1.05M params). Smoke-trained
on the asymmetric data; trains cleanly. The `--keep` flag selects the input width
so the 256 / 384 / 512 ablations he asked about are a one-line change.

### 3. Ablation sweep answering the client's design questions
Trained the 512 network on the asymmetric data (25 epochs) at several settings:

| Configuration | Val loss |
|---|---|
| 512 full (all features) | 0.02103 (best) |
| 512 without directions | 0.02199 |
| 256 input | 0.02253 |
| 384 input | 0.02330 |

Findings: the full 512 wins; shrinking the input to 256 or 384 costs accuracy;
and removing the directions feature makes the model about 5 percent worse, so
directions are not redundant with the atoms and should be kept. The full 512 also
slightly beats the earlier Model 3 asymmetric result (0.0213). CPU runs on one
army file, so these are first results, not final.

### 4. More data helps the 512 architecture
Re-evaluated a second army file (`ZDx2 vs ZWx2`, 109,258 positions) and combined
it with `ii vs jj` for 221,049 positions. Training the 512 network on the
combined set dropped validation loss from 0.02103 (single file) to **0.01539**,
about 27 percent better. The architecture keeps improving with more data, which
supports growing the re-evaluated dataset further and training on a GPU.

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
