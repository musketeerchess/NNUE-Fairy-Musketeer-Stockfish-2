# Milestone 8 — Functional Models

"Functional" means the trained nets **actually play Musketeer**, not merely
train.  There are two senses of it, and this milestone delivers the first fully
and scaffolds the second.

## 1. Functional as a playing agent — DONE, works now

`play.py` turns any trained net into a working move-picker: the private engine
supplies the legal moves and resulting positions, and the net evaluates each and
plays the best (1-ply search on its evaluation).  No engine recompilation is
needed — this plays legal Musketeer chess today.

```bash
# best move for a position, using the strongest net
python play.py --net model3 --fen "*u***h**/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/HU****** w KQkq - 0 1"
# -> [model3] best move: g1h3

# the net plays a whole game against itself
python play.py --net model3 --selfplay --move-cap 120
```

Verified: the nets choose sensible developing moves and play complete games.
The **arena** (`arena.py`) is the multi-net version of the same thing.

### Arena result with the fully-trained nets

Round-robin, 4 games per ordered pair (24 games), 1-ply, move-cap 160:

```
1. model1   7.0/12  (58%)
2. model4   6.0/12  (50%)
3. model2   5.5/12  (46%)
4. model3   5.5/12  (46%)
```

**Honest reading — loss and arena disagree, and here's why.**  The loss screen
ranks **Model 3 best** (val loss 0.0245 vs ~0.033); the 1-ply arena narrowly
favours Model 1.  Two reasons: (a) only 2 of 24 games were decisive — a 1-ply
search plays weakly and draws a lot, so the sample is noisy; (b) low regression
loss (matching the engine's eval) does not automatically translate into better
*move choice at 1 ply*.  The **loss ranking is the more reliable measure of
evaluation quality**; a firm playing-strength verdict needs a **deeper search
(2–3 ply) and more games**, or the engine-native nets below.  Both signals are
reported rather than cherry-picked.

## 2. Functional inside the engine (`.nnue`) — scaffolded, needs the arch decision

To have the **stock Fairy-Stockfish binary** play with a net, the net must be in
the engine's `.nnue` format, which is tied to the exact architecture the engine
was compiled with (a sparse HalfKA transformer + fixed small hidden layers).
Our four models are **custom fully-connected nets** (MK128 input, 256/512-wide
hidden), so the stock binary cannot load them as-is.  Two supported routes:

**A. Export our trained weights (done).** `export_weights.py` writes any
checkpoint to a portable `models/<name>_export.npz` + `.json` (architecture
header) for integration into an engine build that matches this architecture.

```bash
python export_weights.py --ckpt models/model3.pt --out models/model3_export
```

**B. Engine-native net via the official trainer (scaffolded).** Produce a
`.nnue` the stock engine loads by training within `variant-nnue-pytorch` (its
HalfKA architecture) and exporting with its `serialize.py`.  Prerequisites, in
order:

1. **Variant config** — copy `docs/engine_native/variant_musketeer.py` over
   `third_party/variant-nnue-pytorch/variant.py` (sets 8 piece types incl.
   Hawk/Unicorn).  *(provided)*
2. **Training data in binary format** — convert our `.plain`
   (`data/processed/*.plain`) to the trainer's `.binpack` via the compiled
   `training_data_loader` / the engine's converter.
3. **Train** with `train.py` on a GPU (Colab), then `serialize.py model.ckpt
   nn.nnue` → an engine-loadable net.
4. `./engine ... setoption name EvalFile value nn.nnue` and play.

### Why the engine-native `.nnue` can't be produced on our side (verified)

This was investigated to the bottom, not assumed:

1. **The trainer reads only `.bin`.** `lib/nnue_training_data_stream.h` opens
   exactly one format — `BinSfenInputStream` (packed-SFEN binary). It does not
   read our `.plain`/FEN text or `.binpack`.
2. **Nothing available produces that `.bin`.** The private engine has no data
   tooling (`learn` → *"Unknown command"*; no `gensfen`/convert), and the
   packed-SFEN encoding is variant-specific C++ with no Python writer.
3. **No C++ compiler here** (`gcc`/`g++`/`cl` all absent) to build the data
   loader; Colab has one, but blockers 1–2 remain regardless.
4. **The blocking unknown:** a `.nnue` must match the **exact NNUE feature set
   and layer architecture the private Musketeer engine was compiled with**. The
   engine validates this (feature hash + sizes) and **rejects** any net that
   doesn't match. That spec lives only in the client's private build, so a net
   produced without it would be refused by the engine.

**What is needed from the client to finish it** — any one of:
- a Fairy-Stockfish build with `gensfen`/`learn` (or the ready `.bin` training
  data), **plus** the Musketeer feature-set name + layer sizes the engine
  expects; then the path is: `variant_musketeer.py` → build loader on Colab →
  `train.py` (GPU) → `serialize.py` → `.nnue` → `setoption EvalFile`; **or**
- the client generates the `.nnue` on their build using our data/architecture.

Until then, the models are functional via `play.py`/`arena.py` (above), which
need none of this.

## What Milestone 8 delivers / what it waits on

- ✅ `play.py` — the models function as Musketeer-playing agents **now**.
- ✅ Arena run with the real nets + an honest loss-vs-play analysis.
- ✅ `export_weights.py` — portable weight hand-off (Model 3 exported).
- ✅ `variant_musketeer.py` — variant config for the engine-native path.
- ⏳ The engine-loadable `.nnue` itself needs the **three client decisions**
  (layer sizes / input scheme / piece values) and the binary-data + GPU steps
  above; it is not something we can finalise without the architecture being
  fixed, so it is scaffolded end-to-end rather than guessed.
