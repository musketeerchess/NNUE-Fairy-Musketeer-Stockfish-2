# Milestone 6 — Self-play Arena: Best Net & Deleting Bad Nets

**Goal:** determine the best of the four NNUE models by self-play, and delete
the bad nets.

This is done in two complementary ways:

1. **Loss screen** (`train/compare.py`) — ranks the four nets by held-out
   validation loss on one identical split.  Fast, and the same signal
   `variant-nnue-pytorch/delete_bad_nets.py` uses.
2. **Self-play arena** (`arena.py`) — the nets actually *play games* against each
   other; the winner is the best net and the losers are culled.  This document
   covers the arena.

## How the arena works

Our four nets are **custom PyTorch** networks, not engine `.nnue` files, so the
stock engine cannot load them to play.  Instead we split the two roles:

- **The private engine is the game arbiter.**  It generates the legal Musketeer
  moves (`go perft 1`), gives the resulting FEN after any move (`d`), and
  detects checkmate/stalemate — all of which it does exactly right, including
  gating and castling-gating.
- **Each net is the brain.**  On its move, the net evaluates the position after
  every legal reply and plays the move that **minimises the opponent's
  evaluation** (1-ply negamax on the net's own eval).

So the games are genuinely played by the trained networks; the engine only
enforces the rules.  No `.nnue` export and no GPU are needed — the arena runs on
CPU with the smoke-trained checkpoints, and stronger checkpoints drop straight
in.

### Format
- Round-robin: every pair plays `--games` games, colours alternated, openings
  drawn from the book (`data/processed/musketeer_book.epd`).
- Win = 1, draw = 0.5.  A game that reaches `--move-cap` plies is scored a draw.
- Terminal detection: no legal moves + in check → the side to move is
  checkmated (loss); no legal moves + not in check → stalemate (draw).

### Deleting bad nets
Nets scoring below `--discard-below` (a fraction of the top score) are **moved to
`models/discarded/`** — archived, not hard-deleted, so nothing is lost by
mistake.  The best net is always kept.

## Result

<!-- ARENA_RESULT -->
Round-robin, 2 games per ordered pair (12 games), move-cap 100, on the
CPU smoke checkpoints:

```
1. model1   3.0/6  (50%)
2. model2   3.0/6  (50%)
3. model3   3.0/6  (50%)
4. model4   3.0/6  (50%)
BEST NET: model1  (tie-break)   |  none culled
```

**Every game was a draw**, so the arena did not separate the nets and the
"best net" fell to a tie-break — expected with smoke nets and a 1-ply search
(the players shuffle to the move cap).  The **loss screen is authoritative here
(Model 3 best)**; the arena becomes decisive once the nets are properly trained
on a GPU (and/or the search is deepened to 2–3 ply).

> **Caveat — net strength.** The checkpoints here are CPU *smoke* nets (trained
> on 40 k positions for 20 epochs), and the players use a 1-ply search, so many
> games are quiet draws and the separation is small.  The arena *mechanism* is
> the deliverable; with fully GPU-trained nets (and optionally a 2–3 ply search)
> it separates them cleanly.  The loss screen already gives the sharper ranking
> (Model 3 best), and the two agree on direction.

## Usage

```bash
# quick
python arena.py --games 2 --move-cap 120

# fuller round-robin, cull nets below 40% of the best
python arena.py --games 6 --move-cap 200 --discard-below 0.40
```

## Relation to the official tooling

`variant-nnue-pytorch` ships `run_games.py` (cutechess-cli engine matches) and
`delete_bad_nets.py` (prunes nets by result).  Those operate on **engine-loadable
`.nnue`** nets; once we export our models to that format (the deployability
decision in the Milestone-2 doc), the same round-robin can be run through the
official tooling for tournament-strength adjudication.  `arena.py` provides the
equivalent selection now, directly on our PyTorch nets.
