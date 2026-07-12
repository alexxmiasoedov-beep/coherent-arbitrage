#!/usr/bin/env python3
"""Свип параметров калмановской стратегии (сигнал = инновация / скользящий std).

Проверяем, есть ли ХОТЬ ОДНА конфигурация с робастным OOS-эджем, честно показывая
всю сетку (без черри-пикинга одной удачной точки).

    python sweep_kalman.py [prices_pkl]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
from src.kalman import simulate_kalman

BURN_IN = 1000
N_PERIODS = 6
ENTRY_GRID = [1.5, 2.0, 2.5]
DELTA_GRID = [1e-4, 1e-5]

_PRICES: pd.DataFrame | None = None


def _init(prices):
    global _PRICES
    _PRICES = prices


def _work(task):
    (y, x), entry_z, delta = task
    r = simulate_kalman(_PRICES[y].values, _PRICES[x].values, burn_in=BURN_IN,
                        delta=delta, r_var=1e-3, entry_z=entry_z, exit_z=0.5,
                        stop_z=4.0, fee=config.TAKER_FEE,
                        notional=config.CAPITAL_PER_LEG_USDT)
    if not r:
        return None
    tail = r["bar_pnl"][BURN_IN:]
    chunks = np.array_split(tail, N_PERIODS)
    pos = int(sum(1 for c in chunks if c.sum() > 0))
    return {"entry_z": entry_z, "delta": delta, "y": y, "x": x,
            "net_pnl": r["net_pnl"], "n_trades": r["n_trades"],
            "pos_share": pos / N_PERIODS, "bar_pnl": r["bar_pnl"]}


def main():
    prices_pkl = sys.argv[1] if len(sys.argv) > 1 else ".cache/px_1m_14400_80.pkl"
    prices = pd.read_pickle(prices_pkl)
    pairs = list(combinations(prices.columns, 2))
    tasks = [(p, ez, d) for ez in ENTRY_GRID for d in DELTA_GRID for p in pairs]
    n_workers = max(1, (os.cpu_count() or 2))
    print(f"Данные: {prices.shape} | пар: {len(pairs)} | конфигов: {len(ENTRY_GRID)*len(DELTA_GRID)} "
          f"| задач: {len(tasks)} | ядер: {n_workers}", flush=True)

    with Pool(n_workers, initializer=_init, initargs=(prices,)) as pool:
        res = [r for r in pool.imap_unordered(_work, tasks, chunksize=32) if r]

    df = pd.DataFrame(res)
    n = prices.shape[0]
    print(f"\n{'='*78}")
    print(f"{'entry_z':>7} {'delta':>7} | {'OOS PnL':>9} {'Sharpe':>7} {'traded':>7} "
          f"{'+pairs':>7} {'robust':>7}")
    print(f"{'='*78}")
    for ez in ENTRY_GRID:
        for d in DELTA_GRID:
            g = df[(df["entry_z"] == ez) & (df["delta"] == d)]
            if g.empty:
                print(f"{ez:>7} {d:>7} | (нет сделок)")
                continue
            eq = np.zeros(n)
            for _, row in g.iterrows():
                eq += row["bar_pnl"]
            oos = eq[BURN_IN:]
            pnl = oos.cumsum()[-1]
            sharpe = oos.mean() / oos.std() * np.sqrt(config.ANNUALIZE_BARS) if oos.std() > 0 else 0.0
            robust = int(((g["net_pnl"] > 0) & (g["pos_share"] >= 0.60) & (g["n_trades"] >= 5)).sum())
            pos_pairs = int((g["net_pnl"] > 0).sum())
            print(f"{ez:>7} {d:>7} | {pnl:>+9.1f} {sharpe:>+7.2f} {len(g):>7} "
                  f"{pos_pairs:>7} {robust:>7}")
    print(f"{'='*78}")
    print("robust = пар с PnL>0, ≥60% периодов в плюс, ≥5 сделок (реальный кандидат в эдж)")


if __name__ == "__main__":
    main()
