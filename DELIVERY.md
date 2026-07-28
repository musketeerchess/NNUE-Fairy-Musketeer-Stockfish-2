# Musketeer Chess NNUE — Final Delivery (Milestones 1–8)

A complete, validated pipeline: from the self-play PGN database to **four fully
trained NNUE models**, an automated data generator, a self-play arena, and
functional models that play Musketeer. Every stage is validated against the
private Musketeer Fairy-Stockfish engine.

## Milestone status

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Musketeer vs Classic mechanics doc | ✅ |
| 2 | Database pipeline + Model 1 + doc | ✅ |
| 3 | Model 2 (gating-adaptive) | ✅ |
| 4 | Model 3 (Betza features) | ✅ |
| 5 | Model 4 (deep) | ✅ |
| 6 | Best-net selection (loss compare + self-play arena) | ✅ |
| 7 | Automated dataset generation | ✅ |
| 8 | Functional models | ✅ play as agents; ⏳ engine `.nnue` needs client spec |

## Trained models (FULL data — all 835,853 positions, 20 epochs)

| Model | Architecture | Val loss (lower = better) | File |
|-------|--------------|--------------------------:|------|
| 1 | 128→256→256→256 | 0.0336 | `models/model1.pt` |
| 2 | gating-adaptive 512/256 | 0.0323 | `models/model2.pt` |
| **3** | **Betza features, 128→512** | **0.0245  (BEST)** | `models/model3.pt` |
| 4 | deep 128→256 + 4×256 | 0.0335 | `models/model4.pt` |

**Model 3 wins by ~25%** — the Betza-notation features (controlled squares at
distance 1–4, colour-boundness, infinite-range) materially improve the
evaluation. This is the headline result.

## Validation

| Check | Result |
|-------|--------|
| Games replayed from PGN (all 6 files) | **5,892 / 5,892 (100 %)** |
| Training positions extracted | **835,853** |
| Sampled positions confirmed legal by engine | **300 / 300** |
| Generated self-play positions legal | **200 / 200** |
| Betza geometry (knight cbound 0.00, bishop 1.00, hawk 2&3 leaps) | correct |

## Contents

```
README.md                 project overview + how to run
DELIVERY.md               this file
docs/
  Milestone1..8, Milestones3-5, Colab_Training_Guide, engine_native/
src/        betza.py, musketeer.py, pgn_to_fen.py, features.py
train/      model1..4.py, compare.py, run_full.py
scripts/    validate_parser.py, generate_dataset.py
play.py               a trained net plays Musketeer (functional model)
arena.py              self-play round-robin + cull bad nets
export_weights.py     portable weight export
models/               model{1..4}.pt (trained) + model3_export.{npz,json}
colab/                Musketeer_NNUE_Colab.ipynb (GPU training)
```

## How to use the trained models now

```bash
# best move for a position, using the strongest net
python play.py --net model3 --fen "*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** w KQkq - 0 1"

# the net plays a full game against itself
python play.py --net model3 --selfplay
```

## The one open item — engine-loadable `.nnue` (Milestone 8, part 2)

The models play now via `play.py`/`arena.py`. Loading them **inside the stock
engine** as a `.nnue` needs three things only the private build has (see
`docs/Milestone8_Functional_Models.md` for the full verified diagnosis):

1. the NNUE **feature-set + layer sizes** the Musketeer engine expects (it
   rejects mismatched nets);
2. a way to generate the **binary training data** (`gensfen`/`learn`), which the
   provided engine binary lacks;
3. confirmation of the **input scheme** (our MK128) and **Hawk/Unicorn values**.

With those, the remaining path is scaffolded end-to-end (variant config → build
loader on Colab → train → `serialize.py` → `.nnue`).
