#!/usr/bin/env python3
"""Прогон калмановской (динамический хедж) стратегии по кэшированному юниверсу.

Сравнение с наивным подходом: тот в OOS дал суммарно −1182 USDT (271 пара).
Здесь весь ряд после burn-in — честный OOS (фильтр каузален), поэтому гоняем ВСЕ
пары, а не только «коинтегрированных» кандидатов.

    python run_kalman.py [prices_pkl] [entry_z] [delta]
    python run_kalman.py .cache/px_1m_14400_80.pkl 1.5 1e-4
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
from src.kalman import simulate_kalman

BURN_IN = 1000
N_PERIODS = 6           # на сколько кусков резать OOS-участок для проверки устойчивости

_PRICES: pd.DataFrame | None = None
_CFG: dict = {}


def _init(prices: pd.DataFrame, cfg: dict) -> None:
    global _PRICES, _CFG
    _PRICES, _CFG = prices, cfg


def _work(pair: tuple[str, str]) -> dict | None:
    y, x = pair
    ya = _PRICES[y].values
    xa = _PRICES[x].values
    r = simulate_kalman(ya, xa, burn_in=BURN_IN, **_CFG)
    if not r:
        return None
    # устойчивость: режем PnL после burn-in на N_PERIODS кусков, считаем прибыльные
    tail = r["bar_pnl"][BURN_IN:]
    chunks = np.array_split(tail, N_PERIODS)
    pos_periods = int(sum(1 for c in chunks if c.sum() > 0))
    return {
        "y": y, "x": x, "n_trades": r["n_trades"],
        "net_pnl": round(r["net_pnl"], 2),
        "win_rate": round(r["win_rate"], 2),
        "pos_periods": pos_periods,
        "pos_share": round(pos_periods / N_PERIODS, 2),
        "bar_pnl": r["bar_pnl"],
    }


def main() -> None:
    prices_pkl = sys.argv[1] if len(sys.argv) > 1 else ".cache/px_1m_14400_80.pkl"
    entry_z = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    delta = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4

    prices = pd.read_pickle(prices_pkl)
    cfg = {"delta": delta, "r_var": 1e-3, "entry_z": entry_z,
           "exit_z": 0.5, "stop_z": 4.0, "fee": config.TAKER_FEE,
           "notional": config.CAPITAL_PER_LEG_USDT}
    pairs = list(combinations(prices.columns, 2))
    n_workers = max(1, (os.cpu_count() or 2))
    print(f"Данные: {prices.shape[0]} баров × {prices.shape[1]} контрактов | пар: {len(pairs)}")
    print(f"Калман: delta={delta} r_var={cfg['r_var']} entry_z={entry_z} burn_in={BURN_IN} "
          f"(OOS-участок ≈ {prices.shape[0]-BURN_IN} баров)")
    print(f"Гоняю на {n_workers} ядрах...", flush=True)

    with Pool(n_workers, initializer=_init, initargs=(prices, cfg)) as pool:
        res = [r for r in pool.imap_unordered(_work, pairs, chunksize=16) if r]

    if not res:
        print("Ни одна пара не дала сделок.")
        return

    # агрегированный эквити по всему юниверсу
    equity_bars = np.zeros(prices.shape[0])
    for r in res:
        equity_bars += r.pop("bar_pnl")
    oos = equity_bars[BURN_IN:]
    equity = oos.cumsum()
    total_pnl = float(equity[-1])
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity - peak).min())
    sharpe = float(oos.mean() / oos.std() * np.sqrt(config.ANNUALIZE_BARS)) if oos.std() > 0 else 0.0

    df = pd.DataFrame(res)
    robust = df[(df["net_pnl"] > 0) & (df["pos_share"] >= 0.60) & (df["n_trades"] >= 5)]
    robust = robust.sort_values(["pos_share", "net_pnl"], ascending=False).reset_index(drop=True)

    print(f"\n{'='*74}")
    print("КАЛМАН (динамический хедж) — OOS по всему юниверсу")
    print(f"{'='*74}")
    print(f"Суммарный OOS PnL:      {total_pnl:+.1f} USDT   (наивный baseline: −1182.6)")
    print(f"Макс. просадка:         {max_dd:+.1f} USDT")
    print(f"Sharpe (годовой):       {sharpe:+.2f}")
    print(f"Пар с плюсом:           {int((df['net_pnl']>0).sum())} из {len(df)} торговавших")
    print(f"Робастных (≥60% периодов в плюс, ≥5 сделок, PnL>0): {len(robust)}")

    out = prices_pkl.replace("px_", "kalman_").replace(".pkl", ".csv")
    df.drop(columns=[c for c in df.columns if c == "bar_pnl"], errors="ignore").to_csv(out, index=False)
    if not robust.empty:
        print("\nТоп робастных пар:")
        print(robust.head(30).to_string(index=False))
    print(f"\nВсе результаты: {out}")


if __name__ == "__main__":
    main()
