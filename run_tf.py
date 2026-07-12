#!/usr/bin/env python3
"""Калмановская стратегия на произвольном таймфрейме: gross/net/robust/Sharpe в одной
таблице. Главная метрика — соотношение GROSS/FEES: растёт ли эдж на сделку с ростом ТФ.

    python run_tf.py <prices_pkl> [burn_in]
    python run_tf.py .cache/px_1h_3000_80.pkl 300
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

ENTRY_GRID = [1.5, 2.0, 2.5, 3.0]
DELTA = 1e-5
N_PERIODS = 6
_PRICES = None
_BURN = 300


def _init(prices, burn):
    global _PRICES, _BURN
    _PRICES, _BURN = prices, burn


def _work(task):
    (y, x), ez = task
    r = simulate_kalman(_PRICES[y].values, _PRICES[x].values, burn_in=_BURN,
                        delta=DELTA, r_var=1e-3, entry_z=ez, exit_z=0.5, stop_z=4.0,
                        fee=config.TAKER_FEE, notional=config.CAPITAL_PER_LEG_USDT)
    if not r:
        return None
    tail = r["bar_pnl"][_BURN:]
    chunks = np.array_split(tail, N_PERIODS)
    pos = int(sum(1 for c in chunks if c.sum() > 0))
    return {"entry_z": ez, "gross": r["gross_pnl"], "fees": r["total_fees"],
            "net": r["net_pnl"], "trades": r["n_trades"],
            "pos_share": pos / N_PERIODS, "bar_pnl": r["bar_pnl"]}


def main():
    prices_pkl = sys.argv[1]
    burn = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    prices = pd.read_pickle(prices_pkl)
    pairs = list(combinations(prices.columns, 2))
    tasks = [(p, ez) for ez in ENTRY_GRID for p in pairs]
    n = prices.shape[0]
    print(f"{prices_pkl}: {n} баров × {prices.shape[1]} монет | пар {len(pairs)} | "
          f"burn_in={burn} → OOS≈{n-burn} баров | комиссия {config.TAKER_FEE*100:.3f}%/нога\n")

    with Pool(max(1, os.cpu_count() or 2), initializer=_init, initargs=(prices, burn)) as pool:
        res = [r for r in pool.imap_unordered(_work, tasks, chunksize=32) if r]
    df = pd.DataFrame(res)

    print(f"{'entry_z':>7} | {'GROSS':>9} {'FEES':>9} {'NET':>10} {'G/F':>5} "
          f"{'Sharpe':>7} {'trades':>8} {'robust':>6}")
    print("-" * 74)
    for ez in ENTRY_GRID:
        g = df[df["entry_z"] == ez]
        if g.empty:
            print(f"{ez:>7} | (нет сделок)")
            continue
        eq = np.zeros(n)
        for _, row in g.iterrows():
            eq += row["bar_pnl"]
        oos = eq[burn:]
        net = oos.cumsum()[-1]
        sharpe = oos.mean() / oos.std() * np.sqrt(config.ANNUALIZE_BARS) if oos.std() > 0 else 0.0
        gross, fees, tr = g["gross"].sum(), g["fees"].sum(), int(g["trades"].sum())
        robust = int(((g["net"] > 0) & (g["pos_share"] >= 0.60) & (g["trades"] >= 5)).sum())
        gf = gross / fees if fees else 0.0
        print(f"{ez:>7} | {gross:>+9.0f} {fees:>9.0f} {net:>+10.0f} {gf:>5.2f} "
              f"{sharpe:>+7.2f} {tr:>8} {robust:>6}")
    print("-" * 74)
    print("G/F = gross ÷ fees (>1 → эдж перекрывает комиссии). ANNUALIZE_BARS в config "
          f"= {config.ANNUALIZE_BARS} (для 1h корректно; для 1m Sharpe масштабирован иначе).")


if __name__ == "__main__":
    main()
