# Musketeer NNUE — Full-Scale Experiments Runbook

Everything in this project is implemented and validated on CPU with smaller data
chunks. This runbook lets you rerun all of it at full scale on your own GPU
machine. The commands are the same as what produced the results in the paper;
only the budgets grow. The code auto-detects CUDA, so nothing needs changing to
use a GPU.

---

## 0. One-time setup

**Python and PyTorch (with CUDA for the GPU).**
```
python -m pip install --upgrade pip
# CUDA build of PyTorch (pick the cuXXX that matches your driver):
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```
Check the GPU is seen:
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**The code.**
```
git clone https://github.com/musketeerchess/NNUE-Fairy-Musketeer-Stockfish-2
cd NNUE-Fairy-Musketeer-Stockfish-2
```

**The two files that are not in the public repo (you own them):**
- `data/piece_values_cp.json` — the cp piece values (from your spreadsheet).
- `Big Database.zip` — the game database.
Place both in the repo root / `data/` as noted.

**The engine** (for the quality check and any re-evaluation): the
`fairy-stockfish-largeboards` or `musketeer-stockfish` executable, in `engine/`.

---

## 1. Build the training set from the database
```
python scripts/parse_bigdb.py --zip "Big Database.zip" --out data/bigdb
```
Produces `data/bigdb/bigdb.jsonl` (the positions) and `data/bigdb/variants.json`
(the per-file rules). It streams game by game, is crash-safe, and resumes with
`--resume`. On our run this was ~134M positions across 55 armies.

---

## 2. The model comparison (Models 1 to 4, geometry, 512, HalfKP, HalfKA, hybrid)
```
python train/compare_parallel.py --train-positions 20000000 --workers <cores> --dim 256
```
- `--workers` = number of CPU cores for the encoder (the encoder is the
  bottleneck even on a GPU, so set this to your core count).
- On a GPU, raise `--train-positions` well above the CPU runs (20M to 50M+).
- Writes `models/compare_results.json` (ranked val loss, params, size) and the
  per-model checkpoints under `models/bigdb/`.

---

## 3. The experiment studies
Each is one command and writes `models/experiments/<name>.json`.
```
python train/experiments.py --exp arch      --budget 20000000 --workers <cores>
python train/experiments.py --exp relu      --budget 20000000 --workers <cores>
python train/experiments.py --exp models    --budget 20000000 --workers <cores>
python train/experiments.py --exp scaling   --workers <cores>
python train/experiments.py --exp crossarmy --budget 5000000  --crossarmy-n 5 --workers <cores>
```
- `arch` = the ten architectures (dim 256/512/1024 with the head variants).
- `relu` = fixed clipped ReLU vs the learnable asymmetric one.
- `models` = HalfKA vs hybrid vs Ultra.
- `scaling` = 800k vs 1.6M positions (raise both for a GPU version).
- `crossarmy` = the same models trained per army (`--crossarmy-n` sets how many).
- Each run reports val loss, parameter count, and the quantized `.nnue` size.

---

## 4. Train the champion on the full data (the main GPU job)
```
python train/train_champion.py --epochs 2 --workers <cores>
```
Auto-picks the best config from `models/experiments/champion.json` (the
1024-wide hybrid with the learnable activation and the 64/64 head) and trains it
on the full dataset, checkpointing to `models/experiments/champion_FULL.pt`.
`--resume-from <ckpt>` continues; `--max-positions N` bounds the pass.

**Important lesson from the CPU run.** At full width the champion started to
overfit with extended training at a constant learning rate: its training loss
fell well below its validation loss. On a GPU with far more data this is exactly
what capacity is waiting for, but to be safe the full run should add:
- a learning-rate schedule (cosine or step decay),
- early stopping on the validation loss,
- optionally weight decay and/or dropout in the head.
These knobs are small additions to `train/train_champion.py` — say the word and I
will wire them in, or your team can. Watch the train-vs-val gap and stop when
validation stops improving.

Validate any checkpoint:
```
python train/eval_checkpoint.py models/experiments/champion_FULL.pt
```

---

## 5. Database quality check (filter mislabeled games)
```
python scripts/quality_check.py "<games>.pgn" \
    --engine engine/fairy-stockfish-largeboards_x86-64.exe \
    --depth 20 --threshold 400 --out quality_report.csv
```
Flags games whose result contradicts a large engine advantage, walks the last
moves backward at depth, and labels each as `insufficient_material` (an expected
draw, e.g. a lone Unicorn vs a bare king — keep) or `advantage_not_converted`
(the genuinely mislabeled games — filter these out of training).

---

## 6. Suggested order for a full-scale pass
1. Parse the full database (Step 1).
2. Run the quality check on the source games (Step 5) and drop the
   `advantage_not_converted` games.
3. Rerun the comparison and the experiment studies at large budget on the GPU
   (Steps 2 and 3).
4. Train the champion, with the regularization above, on the full filtered data
   (Step 4).
5. Validate with `eval_checkpoint.py`.

---

## Reference: what each result means
- **Validation loss** is the mean squared error of the predicted win probability
  against the blended engine-score / game-result target. Lower is better.
- **Size** is the estimated quantized `.nnue` size; it scales with the transformer
  width (128 MB at 256, 256 MB at 512, 512 MB at 1024).
- All runs are matched (same data, split, budget) so differences are the design
  choice, and every run is resumable.

Questions on any step: happy to walk through it.
