# Milestones 3–5 — Models 2, 3 and 4 (and how we compare them)

This document covers the three remaining network architectures and how they are
ranked.  All three consume the same validated 835 k-position database from
Milestone 2 and share the training objective described there (NNUE loss:
`sigmoid(eval)` regressed against a blend of engine eval and game result).

Quick reference to code:

| Model | File | Encoding | Architecture |
|------|------|----------|--------------|
| Model 2 | `train/model2.py` | MK128, routed by gating | dual-path: 128→512→512→1 / 64→256→256→1 |
| Model 3 | `train/model3.py` | MK128-M3 (Betza features) | 128→512→512→512→1 |
| Model 4 | `train/model4.py` | MK128 | 128→256→256→256→256→256→1 |
| Compare | `train/compare.py` | both | ranks all four on one split |

---

## Milestone 3 — Model 2 (gating-adaptive)

**Spec:** *"128 inputs till gating is finished, then switch back to 64 inputs.
512 neurons for 128 inputs and 256 for 64 inputs. Adapt the number of neurons
related to gating."*

Model 2 is **two sub-networks**, and every position is routed to one of them by
its gating state (`features.gating_finished`):

```
gating NOT finished  →  input(128) → 512 → 512 → 1      (pre-gating path)
gating   finished    →  input( 64) → 256 → 256 → 1      (post-gating path)
```

This is natural for Musketeer: early in the game the two waiting pieces must be
represented (the 64 extra gating features of MK128), so the wider 512-neuron
path is used; once both extra pieces have gated onto the board, that information
is already captured by the 64 board features, so the network switches to the
lighter 64-input / 256-neuron path.  Both paths share the MK128 encoder — the
64-input vector is simply the board plane (`x[:64]`), whose gating plane is
already all-zero post-gating.

On our data the split is about **80 % post-gating / 20 % pre-gating** positions
(gating usually completes in the opening), so most inference takes the fast
path.

---

## Milestone 4 — Model 3 (Betza-derived features)

**Spec:** *"128 inputs in all cases. Inputs: gating squares and additional
features (controlled squares at 1,2,3,4 distance on an empty board at e4;
colour-boundness score; infinite range like queen). 512 neurons for all
layers."*

This is where Betza notation pays off.  For every piece type we compute, **on an
empty board with the piece on e4**, a 7-number geometric fingerprint
(`features.piece_geometry`), all straight from the Betza move generator:

| piece | d1 | d2 | d3 | d4 | total | colour-boundness | infinite |
|-------|---:|---:|---:|---:|------:|-----------------:|:--------:|
| Knight  | 0 | 8 | 0 | 0 |  8 | 0.00 | no  |
| Bishop  | 4 | 4 | 4 | 1 | 13 | 1.00 | yes |
| Rook    | 4 | 4 | 4 | 2 | 14 | 0.43 | yes |
| Queen   | 8 | 8 | 8 | 3 | 27 | 0.70 | yes |
| King    | 8 | 0 | 0 | 0 |  8 | 0.50 | no  |
| **Hawk** (ADGH)   | 0 | 8 | 8 | 0 | 16 | 0.75 | no  |
| **Unicorn** (NC)  | 0 | 8 | 8 | 0 | 16 | 0.50 | no  |
| Amazon (QN) | 8 | 16 | 8 | 3 | 35 | 0.54 | yes |

These match the pieces' true nature — the knight is not colour-bound (0.00), the
bishop is perfectly colour-bound (1.00), the Hawk leaps only to distances 2 and
3, the Amazon is the most powerful.  The **128-input Model-3 vector** is:

```
[  0: 64]  board material plane (signed, side-to-move oriented)
[ 64:120]  8 piece types × 7 geometry features, each × signed on-board count
[120:128]  gating presence per file (own +1 / opponent −1)
```

so the network sees not merely *"a piece is here"* but *"a piece with this
mobility / colour-boundness / range is here, and there are N more of them than
the opponent."*  The tower is 512-wide throughout, per spec.

---

## Milestone 5 — Model 4 (deep 256-wide)

**Spec:** *"128 inputs → 256 neurons, then four additional hidden layers of 256
neurons each; final layer outputs the evaluation."*

```
input(128, MK128) → 256 → 256 → 256 → 256 → 256 → 1
```

Same MK128 encoding as Model 1, but a deeper tower (five 256-wide layers).  This
tests whether depth alone — without engineered features — improves the
evaluation.

---

## Milestone 6 (first cut) — comparing the four

`train/compare.py` trains all four models under **matched conditions** — the
same positions, the same held-out 10 % validation split (fixed seed), the same
epochs and optimiser — and ranks them by validation loss.  A matched run
(60 k positions, identical 10 % val split, 15 epochs, seed 0) gives:

| rank | model | val loss | params | note |
|:----:|-------|---------:|-------:|------|
| 1 | **Model 3** (Betza features) | **0.03539** | 591,873 | best — engineered features add real signal |
| 2 | Model 2 (gating-adaptive)    | 0.03970 | 411,906 | routing helps |
| 3 | Model 1 (256 baseline)       | 0.04206 | 164,865 | baseline |
| 4 | Model 4 (deep 256)           | 0.04286 | 296,449 | depth alone doesn't help at this scale |

Model 3's Betza-derived features win clearly; the deep Model 4 needs more data,
epochs and regularisation before extra depth pays off.  These are CPU smoke
numbers — a full GPU run over all 835 k positions will separate the models
further and feed the true engine arena.

> **Note on the "arena".** Validation loss is a sound first screen and is the
> same signal `variant-nnue-pytorch/delete_bad_nets.py` uses to prune weak nets.
> The *full* Milestone-6 arena plays the trained nets against each other inside
> Fairy-Stockfish, which requires exporting each PyTorch net to the engine's
> `.nnue` format.  That step depends on the deployability decision flagged in the
> Milestone-2 doc (256-wide contract spec vs the engine's fixed layer sizes) and
> on the GPU-trained final nets, and is wired up in `arena.py` to run once those
> are in place.

---

## Reproduce

```bash
python train/model2.py --data data/processed/train1.jsonl --limit 40000 --epochs 20 --lr 3e-3 --out models/model2_smoke.pt
python train/model3.py --data data/processed/train1.jsonl --limit 40000 --epochs 20 --lr 3e-3 --out models/model3_smoke.pt
python train/model4.py --data data/processed/train1.jsonl --limit 40000 --epochs 20 --lr 3e-3 --out models/model4_smoke.pt
python train/compare.py --data data/processed/train1.jsonl --limit 60000 --epochs 15
```
