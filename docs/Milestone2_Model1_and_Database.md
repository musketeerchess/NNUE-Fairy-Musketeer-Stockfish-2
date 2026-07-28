# Milestone 2 — From the Pre-configured Database to the First NNUE (Model 1)

**Project:** Musketeer Chess NNUE.  **Deliverables:** (1) this document, and
(2) the first NNUE model developed from the pre-configured database.

Model 1, per the contract:

> *"First layer must have … 128 inputs, these inputs should be transformed with
> 256 neurons.  Then next hidden layers will use 256 neurons."*

This document explains, end to end, how the self-play database becomes training
data, how a position is turned into the 128 network inputs, the exact Model-1
architecture, how it is trained, and the results — written so that someone
without a programming background can reproduce it by running the listed
commands.

---

## 1. The pipeline at a glance

```
 self-play PGN            src/pgn_to_fen.py          src/features.py         train/model1.py
(SAN moves + evals)  ─►  (replay each move,     ─►  (128-input encoding  ─►  (128→256→256→256→1
                          emit FEN+eval+result)      per position)             NNUE, trained)
```

Every step is validated against the private Musketeer Fairy-Stockfish engine.

---

## 2. From PGN to training records

The database is thousands of high-quality self-play games in **PGN**, each move
annotated with the engine's evaluation, e.g.

```
1. exd5 {+1.20/18} e6 {-1.28/18 4} 2. dxe6 {+1.51/18 0.7} Bxe6 {-1.28/18 0.1} …
```

`src/pgn_to_fen.py` reads each game's `VariantMen` header (the Betza rules of
every piece) and its start `FEN`, then **replays every SAN move** through a
Musketeer board (`src/musketeer.py`), which correctly handles the mechanics from
Milestone 1: 10-rank FEN, gating (`Nc3/U`), castling-gating (`O-O/He`), the
check-forfeit rule, en passant and per-side promotions.

For each move we emit one training record:

| field        | meaning                                                        |
|--------------|----------------------------------------------------------------|
| `fen`        | the position **before** the move (10-rank Musketeer FEN)       |
| `score_cp`   | the engine eval of that position, in centipawns, **from the side-to-move's view** (`{+1.20/18}` → `+120`; mate → ±30000) |
| `move`       | the move played (SAN)                                          |
| `ply`        | half-move index within the game                               |
| `result_stm` | final game result from the side-to-move's view (+1 win / 0 draw / −1 loss); the PGN's *"False draw claim"* comments are ignored, we trust the header `Result` |

**Validation (whole training set):**

- **5,892 / 5,892 games replayed with zero errors** across all six
  `uni hawk training nnue-*.pgn` files.
- **835,853 training positions** produced.
- **300 / 300 sampled positions** (including deep endgames) confirmed by the
  engine to be legal, well-formed Musketeer positions
  (`scripts/validate_parser.py`).

Outputs live in `data/processed/` in two formats: our `.jsonl` (used by the
trainer below) and the Stockfish/Fairy `.plain` text format (for the official
`variant-nnue-pytorch` trainer, used later for the deployable nets).

---

## 3. The 128 inputs — the **MK128** encoding

A neural net needs numbers, not a FEN.  We turn each position into **128 input
values**, matching the contract's "128 inputs … the first input is 64 features":

```
MK128 = 64 board features + 64 gating features
```

- **Board plane (inputs 0–63)** — one value per board square, seen **from the
  side to move** (the board is mirrored for Black so the network always sees its
  own pieces moving up the board).  The value is the piece's material value,
  **positive for our pieces, negative for the opponent's**, scaled to about
  ±1.  These are "the 64 features."

- **Gating plane (inputs 64–127)** — for every extra piece still **waiting** to
  be gated (Hawk/Unicorn on rank 0/9), a value is written on the square it will
  gate onto, again + for us / − for the opponent.  When all extra pieces have
  been gated this plane becomes all-zero, so the position is effectively
  described by 64 features again.

That last property is deliberate: it is exactly the *"128 inputs until gating is
finished, then switch back to 64"* behaviour that **Model 2** will formalise,
so Model 1 and Model 2 share one encoding.

> **Design note for review:** MK128 uses one *value* per square (material +
> position), which is simple and trains well as a first model.  Richer options —
> the sparse HalfKA feature set the official trainer uses, or the client's
> 8-bit `piece_mapping` per square — are drop-in replacements for
> `src/features.py::encode_fen` and are planned for Models 3–4 (which the
> contract specifies use extra Betza-derived features). Piece values for Hawk/
> Unicorn (700/550 cp) are first estimates and easy to tune.

---

## 4. Model 1 — architecture

`train/model1.py`:

```
input (128)
  → Linear(128, 256)  → clipped-ReLU        (the feature transformer: "128 inputs
                                              transformed with 256 neurons")
  → Linear(256, 256)  → clipped-ReLU        (hidden layer, 256 neurons)
  → Linear(256, 256)  → clipped-ReLU        (hidden layer, 256 neurons)
  → Linear(256, 1)                          = evaluation (side-to-move, cp)
```

- **Clipped ReLU** (`clamp(x, 0, 1)`) is the standard NNUE activation — it keeps
  activations in the range the quantised engine evaluation uses, so the net can
  later be exported to the integer format Fairy-Stockfish loads.
- ~165k parameters; the number of 256-wide hidden layers is a command-line knob
  (`--hidden`).

> **Deployability note for review:** the official Fairy-Stockfish NNUE keeps its
> hidden layers small (16 then 32 neurons) for speed, whereas the contract asks
> for 256-wide hidden layers.  Model 1 here follows the **contract's** 256-wide
> spec as a standalone PyTorch network.  Producing a net the *stock* engine can
> load without changes would instead use the official trainer's layer sizes.
> Which of the two we want for the engine-playable nets (Milestone 6) is the one
> open question to confirm — both paths are wired up in this repo.

---

## 5. Training objective and procedure

We use the **standard NNUE loss**: the network's evaluation is squashed to a win
probability and regressed against a blend of the engine's own evaluation and the
actual game outcome:

```
pred   = sigmoid( eval  / 361 )
target = λ · sigmoid( score_cp / 361 )  +  (1 − λ) · wdl_result
loss   = mean( (pred − target)² )
```

- `λ = 0.7` blends 70 % "trust the engine eval" with 30 % "trust the game
  result" (a common, robust default; tunable via `--lam`).
- Optimiser Adam, clipped-ReLU activations, 5 % of positions held out for
  validation.

### Result (CPU smoke run, 40 000 positions, 20 epochs)

| epoch | train loss | val loss |
|------:|-----------:|---------:|
| 1     | 0.0533     | 0.0522   |
| 10    | 0.0474     | 0.0484   |
| 20    | **0.0413** | **0.0432** |

Train and validation loss fall together every epoch — the network is learning a
genuine Musketeer evaluation and generalising (no over-fitting).  This
establishes the working Model-1 pipeline; a full run over all 835 k positions on
a GPU (Milestone-6 territory) will push the loss substantially lower and produce
the net used in the model comparison.

---

## 6. How to reproduce (copy-paste)

Prepare the data (already done; regenerates `data/processed/`):

```bash
python src/pgn_to_fen.py "uni hawk training nnue-1.pgn" \
    --jsonl data/processed/train1.jsonl --plain data/processed/train1.plain
```

Sanity-check the parser against the engine:

```bash
python scripts/validate_parser.py data/processed/train1.jsonl --n 300
```

Train Model 1 (quick CPU smoke test):

```bash
python train/model1.py --data data/processed/train1.jsonl --limit 40000 \
    --epochs 20 --lr 3e-3 --out models/model1_smoke.pt
```

Full training (all files; use a CUDA GPU for speed):

```bash
python train/model1.py --data "data/processed/uni_hawk_training_nnue-*.jsonl" \
    --epochs 30 --out models/model1.pt
```

---

## 7. Status & what Milestone 2 delivers

- ✅ Database → FEN/eval/result pipeline, **validated** (100 % replay, engine-checked).
- ✅ 128-input **MK128** feature encoding.
- ✅ **Model 1** implemented to spec (128 → 256 → 256 → 256 → 1) and **trains /
  converges**.
- ✅ This document.

**To confirm with you (Zied):** (a) engine-deployable layer sizes vs the 256-wide
hidden spec; (b) whether MK128 matches your intended 128-input scheme or you want
the sparse HalfKA / 8-bit `piece_mapping` variant; (c) Hawk/Unicorn piece values.
These feed directly into Models 2–4.
