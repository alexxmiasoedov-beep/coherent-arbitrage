#!/usr/bin/env python3
"""Разложение PnL калмановской стратегии на gross (до комиссий) и net.

Отвечает на вопрос: сигнала нет вовсе (gross≈0) или сигнал есть, но издержки съедают
(gross>0, net<0)? Во втором случае осмысленно уходить на больший таймфрейм.

    python gross_vs_net.py [prices_pkl]
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from itertools import combinations
from multiprocessing import Pool

import pandas as pd

import config
from src.kalman import simulate_kalman

ENTRY_GRID = [1.5, 2.0, 2.5]
DELTA = 1e-5
_PRICES = None


def _init(prices):
    global _PRICES
    _PRICES = prices


def _work(task):
    (y, x), ez = task
    r = simulate_kalman(_PRICES[y].values, _PRICES[x].values, burn_in=1000,
                        delta=DELTA, r_var=1e-3, entry_z=ez, exit_z=0.5, stop_z=4.0,
                        fee=config.TAKER_FEE, notional=config.CAPITAL_PER_LEG_USDT)
    if not r:
        return None
    return {"entry_z": ez, "gross": r["gross_pnl"], "fees": r["total_fees"],
            "net": r["net_pnl"], "trades": r["n_trades"]}


def main():
    prices_pkl = sys.argv[1] if len(sys.argv) > 1 else ".cache/px_1m_14400_80.pkl"
    prices = pd.read_pickle(prices_pkl)
    pairs = list(combinations(prices.columns, 2))
    tasks = [(p, ez) for ez in ENTRY_GRID for p in pairs]
    with Pool(max(1, os.cpu_count() or 2), initializer=_init, initargs=(prices,)) as pool:
        res = [r for r in pool.imap_unordered(_work, tasks, chunksize=32) if r]
    df = pd.DataFrame(res)
    print(f"Юниверс: {prices.shape[1]} монет, {len(pairs)} пар, delta={DELTA}, "
          f"комиссия {config.TAKER_FEE*100:.3f}%/нога\n")
    print(f"{'entry_z':>7} | {'GROSS':>10} {'-FEES':>10} {'= NET':>10} {'trades':>8} "
          f"{'fee/trade':>9}")
    print("-" * 62)
    for ez in ENTRY_GRID:
        g = df[df["entry_z"] == ez]
        gross, fees, net, tr = g["gross"].sum(), g["fees"].sum(), g["net"].sum(), g["trades"].sum()
        print(f"{ez:>7} | {gross:>+10.1f} {fees:>10.1f} {net:>+10.1f} {int(tr):>8} "
              f"{fees/tr:>9.3f}")
    print("-" * 62)
    print("gross≈0 → сигнала нет; gross>0 & net<0 → сигнал есть, но комиссии съедают.")


if __name__ == "__main__":
    main()
