"""
Elo tournament for the NNUE models.

Reuses the arena's engine arbiter, net players, and game player. Plays a fixed
number of games per ordered pair from the opening book (colours alternate),
records win / draw / loss, and fits Elo ratings by maximum likelihood under the
logistic model, anchored so the mean rating is zero, with an approximate
standard error for each rating.

This is the same harness at any scale: run it here on the small phase-one models
to see the Elo gaps, or point it at a larger set of models with more games on a
bigger machine.

Usage:
    python elo_arena.py --games 20 --move-cap 120 --depth 1 --out elo_report.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena   # Arbiter, NetPlayer, load_players, play_game, ENGINE, VARIANT_MEN


def fit_elo(names, pair):
    """pair[(i,j)] = (score_i, games) with i<j; score_i = wins_i + 0.5*draws.
    Returns (ratings, standard_errors), ratings anchored to mean 0."""
    n = len(names)
    R = [0.0] * n
    lr = 8.0
    for _ in range(40000):
        grad = [0.0] * n
        for (i, j), (s_i, g) in pair.items():
            if g == 0:
                continue
            p = 1.0 / (1.0 + 10 ** ((R[j] - R[i]) / 400.0))   # expected score of i
            d = (math.log(10) / 400.0) * (s_i - g * p)
            grad[i] += d
            grad[j] -= d
        step = max(abs(x) for x in grad)
        for i in range(n):
            R[i] += lr * grad[i]
        m = sum(R) / n
        R = [r - m for r in R]
        if step < 1e-9:
            break
    # approximate SE from Fisher information
    info = [0.0] * n
    for (i, j), (s_i, g) in pair.items():
        p = 1.0 / (1.0 + 10 ** ((R[j] - R[i]) / 400.0))
        fi = (math.log(10) / 400.0) ** 2 * g * p * (1 - p)
        info[i] += fi
        info[j] += fi
    se = [(1.0 / math.sqrt(v)) if v > 1e-12 else float("nan") for v in info]
    return R, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10, help="games per ordered pair")
    ap.add_argument("--book", default="data/processed/musketeer_book.epd")
    ap.add_argument("--move-cap", type=int, default=120)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="elo_report.json")
    args = ap.parse_args()

    # point the arbiter at whichever engine build is present
    if not os.path.exists(arena.ENGINE):
        for alt in ("fairy-stockfish-largeboards_x86-64-bmi2.exe",
                    "fairy-stockfish-largeboards_x86-64-modern.exe"):
            p = os.path.join(os.path.dirname(arena.ENGINE), alt)
            if os.path.exists(p):
                arena.ENGINE = p
                break

    random.seed(args.seed); torch.manual_seed(args.seed)
    book = [l.strip() for l in open(args.book, encoding="utf-8") if l.strip()]
    players = arena.load_players()
    for pl in players:
        pl.depth = args.depth
    if len(players) < 2:
        raise SystemExit("need at least two models to rate")
    names = [p.name for p in players]
    print(f"engine: {os.path.basename(arena.ENGINE)}", flush=True)
    print(f"players: {', '.join(names)}  |  {args.games} games per ordered pair",
          flush=True)

    arb = arena.Arbiter()
    wins = {n: 0.0 for n in names}
    pair = {}
    for a, b in itertools.combinations(range(len(players)), 2):
        s_a = 0.0; g = 0
        for k in range(args.games):
            start = random.choice(book)
            white, black = (players[a], players[b]) if k % 2 == 0 else (players[b], players[a])
            r = arena.play_game(arb, white, black, start, args.move_cap)  # white POV
            # convert to player-a score
            if white is players[a]:
                sa = 0.5 * (r + 1)          # r in {-1,0,1} -> {0,0.5,1}
            else:
                sa = 0.5 * ((-r) + 1)
            s_a += sa; g += 1
            wins[names[a]] += sa; wins[names[b]] += (1 - sa)
        pair[(a, b)] = (s_a, g)
        print(f"  {names[a]} vs {names[b]}: {s_a:.1f}/{g}", flush=True)

    R, se = fit_elo(names, pair)
    order = sorted(range(len(names)), key=lambda i: -R[i])
    print("\n" + "=" * 52)
    print(f"{'model':<18}{'Elo':>8}{'+/-':>7}{'score':>10}")
    print("-" * 52)
    total_games = sum(g for _, g in pair.values()) * 2 / len(names)
    for i in order:
        print(f"{names[i]:<18}{R[i]:>8.0f}{se[i]:>7.0f}{wins[names[i]]:>10.1f}")
    print("=" * 52)
    out = {"engine": os.path.basename(arena.ENGINE), "games_per_pair": args.games,
           "depth": args.depth,
           "ratings": [{"model": names[i], "elo": round(R[i], 1),
                        "se": round(se[i], 1), "score": round(wins[names[i]], 1)}
                       for i in order]}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
