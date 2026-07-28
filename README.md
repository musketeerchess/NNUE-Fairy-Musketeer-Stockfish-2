# Musketeer Chess — NNUE Training Project

Train NNUE evaluation networks for **Musketeer Chess** (Hawk + Unicorn) using a
custom Fairy-Stockfish and the [`variant-nnue-pytorch`](https://github.com/fairy-stockfish/variant-nnue-pytorch)
trainer. Rebuilt from scratch.

## Layout

```
engine/            private Musketeer Fairy-Stockfish binaries (ground-truth validator)
data/
  raw/             source PGNs (self-play games w/ inline evals)
  processed/       parser output: (FEN, eval_cp, result) -> .plain -> .binpack
src/
  betza.py         Betza-notation move generator            [DONE, validated]
  musketeer.py     10-rank board, FEN, SAN apply, gating    [next]
  pgn_to_fen.py    PGN -> training positions                [next]
  features.py      NNUE feature encoders (per model)        [later]
train/             4 NNUE model configs + training driver   [later]
arena.py           self-play round-robin, cull bad nets     [M6]
docs/              milestone documents
third_party/
  variant-nnue-pytorch/   cloned trainer (reads .bin/.binpack, needs CUDA GPU)
```

## Pipeline

```
PGN (SAN + evals)                    the 6 "uni hawk training nnue-*.pgn" files
  │  src/pgn_to_fen.py  (replays moves via src/betza.py + src/musketeer.py)
  ▼
(FEN, eval_cp, side, result) records   ← validated square-by-square vs engine/
  │  -> .plain text format
  ▼
.binpack  (Fairy-Stockfish training format)
  │  third_party/variant-nnue-pytorch/train.py  (variant.py extended: 8 piece types)
  ▼
trained NNUE  ×4 architectures  ->  arena.py picks the best  (M5/M6)
```

## Data on hand

- 6 × `uni hawk training nnue-*.pgn` (~982 games each, ≈5,900+ games) — training.
- 4 × `uni hawk testset 60mov-1min *.pgn` + suite — evaluation.
- `uh train nnue Endgames & Middlegames*.pgn` — endgame/middlegame set.
- All headers carry `VariantMen` (Betza piece rules) and inline evals `{+1.20/18}`.

## Milestones

1. ✅ Mechanics doc — `docs/Milestone1_Musketeer_vs_Classic.md`
2. ✅ Model 1 (128→256→256→256→1) + database pipeline + doc
   — `docs/Milestone2_Model1_and_Database.md`, `train/model1.py`, `src/features.py`
3. ✅ Model 2: gating-adaptive (512@128-in / 256@64-in) — `train/model2.py`
4. ✅ Model 3: 128→512 flat + engineered Betza features — `train/model3.py`
5. ✅ Model 4: 128→256 then 4×256 hidden — `train/model4.py`
   — Models 2–4 doc: `docs/Milestones3-5_Models2-4.md`
6. ✅ Best-net selection: loss compare (`train/compare.py`) + **self-play arena**
   (`arena.py`: engine arbiter + nets as brains, round-robin, culls losers)
   — doc: `docs/Milestone6_Arena.md`
7. ✅ Automate future datasets — engine self-play w/ evals — `scripts/generate_dataset.py`
   — doc: `docs/Milestone7_Dataset_Automation.md`
8. ◑ Functional models: `play.py` (nets play Musketeer now) + `export_weights.py`
   + engine-native scaffold (`docs/engine_native/variant_musketeer.py`).
   Engine-loadable `.nnue` waits on client's arch decision. — doc: `docs/Milestone8_Functional_Models.md`

**Trained models (full 835k data, 20 epochs):** model1 val 0.0336 · model2 0.0323 ·
**model3 0.0245 (best)** · model4 0.0335 → `models/model{1,2,3,4}.pt`

## Toolchain status

- Engine runs; `musketeer` variant confirmed; parses rank-0/9 waiting FENs. ✅
- `pyffish` installs on Python 3.14 (no built-in `musketeer` → we use our own
  Betza engine as the mover, engine as validator). ✅
- Trainer cloned. **Needs a CUDA GPU** for the actual training runs.
