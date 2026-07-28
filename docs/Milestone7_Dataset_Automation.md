# Milestone 7 — Automating the Pre-configured Database

**Goal:** generate the NNUE training database automatically, so future models
(any piece combination) no longer depend on hand-provided PGNs.

**Approach:** follow `chess-variant-stats` (engine self-play from an opening book
→ positions), but drive the **private Musketeer engine directly**.  Two reasons:

1. The Musketeer variant is **not in public pyffish**, which `chess-variant-stats`
   requires — so its `generate_games.py` cannot play Musketeer.
2. `chess-variant-stats` records only the best move and result; **NNUE training
   needs the engine's evaluation per position**, which our generator captures.

## What it does

`scripts/generate_dataset.py`:

1. Loads an opening book of start FENs (`data/processed/musketeer_book.epd`;
   here extracted from the existing games, but any book works — e.g. from
   [ianfab/books](https://github.com/ianfab/books) or the bookgen tool).
2. For each game, has the engine **self-play** move by move at a fixed
   `movetime`, reading each position's FEN (via `d`) and the engine's
   side-to-move score (via the `go` info lines).
3. Detects terminal states (checkmate → loss for side to move; stalemate/cap →
   draw) and writes one record per ply in the **same schema as the PGN parser**:
   `(fen, score_cp, move, ply, stm, result_stm)` → JSONL **and** `.plain`.

Because the output format is identical to `src/pgn_to_fen.py`, generated data
feeds straight into `train/model{1..4}.py` with no changes.

## Verified

A 3-game smoke run (movetime 60 ms) self-played two decisive games (checkmate
detected) and one drawn (move cap), producing **532 positions**; a 200-position
sample was **100 % legal** under the engine (`scripts/validate_parser.py`),
confirming the generated FENs and schema are correct.

## Usage

```bash
python scripts/generate_dataset.py --count 20 --movetime 100 \
    --book data/processed/musketeer_book.epd \
    --jsonl data/processed/selfplay.jsonl --plain data/processed/selfplay.plain
```

Scale up `--count` and `--movetime` (and use a larger opening book) for a real
database — the client's note of "≥100 k positions" is reached by raising
`--count`.  For a different piece combination, supply a book with those pieces
and the matching engine variant configuration; the rest of the pipeline is
unchanged.

## Extending

- **Parallelism:** run several instances with different `--seed` and concatenate
  the JSONL files (mirrors `chess-variant-stats --workers`).
- **Other pieces:** the private engine plays Musketeer with the default Hawk/
  Unicorn out of the box; other extra pieces need the engine's variant config
  (Betza `VariantMen`), then the same generator applies.
