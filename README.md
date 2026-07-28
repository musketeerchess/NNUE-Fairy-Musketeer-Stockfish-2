# Musketeer Chess NNUE

Training and evaluation of NNUE (efficiently updatable neural network) evaluation
functions for **Musketeer Chess** — chess with two additional "Musketeer" pieces
(here **Hawk** and **Unicorn**) that are gated onto the board during play.

The project provides a full, self-contained pipeline: it parses self-play games
into training data, encodes positions into network features (including features
derived from Betza movement notation), trains four network architectures,
selects the strongest by self-play, and lets you play against a trained model
from the command line or a web board. Move legality and positions are validated
against a Musketeer Fairy-Stockfish build.

## Repository layout

```
src/
  betza.py         Betza-notation move generator
  musketeer.py     10-rank board: FEN, SAN application, gating, castling-gating
  pgn_to_fen.py    PGN -> (FEN, evaluation, result) training records
  features.py      Feature encodings (MK128 + Betza-derived geometry)
train/
  model1.py        128 -> 256 -> 256 -> 256 -> 1
  model2.py        gating-adaptive: 128->512 pre-gating / 64->256 post-gating
  model3.py        128 -> 512 flat, with engineered Betza features
  model4.py        deep: 128 -> 256 + four 256-wide hidden layers
  compare.py       rank all four on one held-out split
  run_full.py      train all models on the full dataset
scripts/
  validate_parser.py   cross-check parsed positions against the engine
  generate_dataset.py  produce fresh training data via engine self-play
play.py            best move / self-play for a trained model
play_human.py      play a game move-by-move against a model
webplay.py         browser board to play a model (with placement phase)
arena.py           self-play round-robin between models; prunes weak nets
export_weights.py  export a trained model to a portable .npz + .json
```

Not tracked in this repository (large or environment-specific): the engine
binaries (`engine/`), the PGN game database and parsed data (`data/`), the
trained model files (`models/`), and the cloned trainer (`third_party/`).

## Pipeline

```
self-play PGN (SAN moves + evaluations)
    |  src/pgn_to_fen.py  (replays moves via betza.py + musketeer.py)
    v
(FEN, evaluation, result) records   (validated square-by-square vs the engine)
    |  src/features.py  (128-input encodings)
    v
four NNUE architectures  ->  compare.py / arena.py select the best
    |  export_weights.py
    v
portable weights  /  engine-native .nnue (via variant-nnue-pytorch)
```

## Feature encodings

- **MK128** — 128 inputs: 64 board features (signed, side-to-move oriented) plus
  64 gating features for the pieces still waiting to enter. The gating half
  becomes zero once both extra pieces have gated, so the position is then
  described by 64 features.
- **Betza geometry** (used by `model3`) — per-piece descriptors computed from
  the Betza definition on an empty board: number of controlled squares at
  distance 1–4, colour-boundness, and infinite-range detection.

## Models and results

Trained on the full dataset (about 835k positions), 20 epochs. Lower validation
loss is better.

| Model  | Architecture                          | Val loss |
|--------|---------------------------------------|---------:|
| model1 | 128 -> 256 -> 256 -> 256              | 0.0336   |
| model2 | gating-adaptive (512 / 256)           | 0.0323   |
| model3 | 128 -> 512, Betza features            | **0.0245** |
| model4 | 128 -> 256 + four 256-wide layers     | 0.0335   |

`model3`, which uses the engineered Betza features, gives the best evaluation.

## Usage

Parse games into training records:

```bash
python src/pgn_to_fen.py "uni hawk training nnue-1.pgn" \
    --jsonl data/processed/train1.jsonl --plain data/processed/train1.plain
```

Validate parsed positions against the engine:

```bash
python scripts/validate_parser.py data/processed/train1.jsonl --n 300
```

Train a model (auto-detects a CUDA GPU; falls back to CPU):

```bash
python train/model3.py --data "data/processed/*.jsonl" --epochs 30 --out models/model3.pt
```

Compare all four, or run the self-play arena:

```bash
python train/compare.py --data "data/processed/*.jsonl" --epochs 20
python arena.py --games 6 --move-cap 200
```

Play against a trained model:

```bash
python webplay.py --net model3        # then open http://localhost:8000
python play.py --net model3 --fen "<FEN>"
```

## Requirements

- Python 3.9+ with `torch` and `numpy`.
- A Musketeer Fairy-Stockfish build (used as move/position oracle and validator).
- A CUDA GPU is recommended for full-scale training; all code runs on CPU.

## Generating fresh data

`scripts/generate_dataset.py` drives the engine to self-play games from an
opening book and records each position with its evaluation, in the same format
as the PGN parser — so new datasets (for other piece combinations) can be
produced without hand-collected games.
