# Training on Google Colab (free GPU)

The training scripts auto-detect CUDA, so they run on Colab's GPU with no code
changes. You need two files (both provided):

- `Musketeer_NNUE_Colab.ipynb` — the notebook
- `Musketeer_Colab_bundle.zip` — the code + the 6 self-play PGNs (6.4 MB)

## Steps

1. Go to **https://colab.research.google.com** → **File → Upload notebook** →
   choose `Musketeer_NNUE_Colab.ipynb`.
2. **Runtime → Change runtime type → Hardware accelerator: GPU → Save.**
3. Run the cells top to bottom:
   - **Step 1** confirms the GPU (e.g. Tesla T4).
   - **Step 2** — click *Choose Files* and upload `Musketeer_Colab_bundle.zip`.
   - **Step 3** rebuilds the 835k-position dataset from the PGNs (pure Python,
     a couple of minutes — the engine is **not** needed for parsing).
   - **Step 4** trains all four models on the GPU. The first model encodes and
     caches the dataset (`.npz`); the rest reuse it. `EPOCHS` is adjustable.
   - **Step 5** compares the four on one split (Model 3 should win).
   - **Step 6** downloads `trained_models.zip` (or copy it to Drive).

## Notes

- **Data transfer is small** because we upload the 21 MB of PGNs (6.4 MB zipped)
  and regenerate the 124 MB dataset inside Colab, rather than uploading it.
- On a T4, encoding is the slow part (CPU, ~10–15 min one-time, cached); each
  model's GPU training is fast. Increase `EPOCHS`/`BATCH` freely on GPU.
- To persist across sessions, use the Google Drive option in Step 2 and the
  `!cp ... /content/drive/MyDrive/` line in Step 6.
- Free Colab has session time limits; if disconnected, the cached `.npz` (if on
  Drive) lets you resume training quickly.
