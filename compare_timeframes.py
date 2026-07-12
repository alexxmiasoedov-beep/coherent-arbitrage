#!/usr/bin/env python3
"""Сравнение таймфреймов сигнала на ОДИНАКОВОМ календарном окне.

Для каждого интервала грузим столько баров, сколько укладывается в WINDOW_DAYS,
делим 70/30 (formation/test) и прогоняем один и тот же бэктест. Смотрим, где лучше
Sharpe / просадка / переторговля. Решают данные, а не рассуждения.

    python compare_timeframes.py
"""
from __future__ import annotations

import pandas as pd

from src import data
from src.backtest import run

WINDOW_DAYS = 14
TOP_N = 15                       # компактная вселенная — быстрее и меньше экзотики
INTERVALS = {                    # интервал -> минут в баре
    "1h": 60, "15m": 15, "5m": 5, "1m": 1,
}


def main() -> None:
    symbols = data.top_symbols(TOP_N)
    print(f"Вселенная: {len(symbols)} контрактов, окно {WINDOW_DAYS} дней.\n")

    rows = []
    for interval, minutes in INTERVALS.items():
        n_bars = WINDOW_DAYS * 24 * 60 // minutes
        print(f"[{interval}] гружу {n_bars} баров/символ...")
        prices = data.price_matrix_n(symbols, interval, n_bars)
        if prices.shape[1] < 3 or prices.shape[0] < 50:
            print(f"  недостаточно данных ({prices.shape}), пропуск.")
            continue
        form_bars = int(prices.shape[0] * 0.7)
        res = run(prices=prices, form_bars=form_bars, verbose=False)
        rows.append({
            "interval": interval,
            "bars": prices.shape[0],
            "pairs": res["pairs_traded"],
            "trades": res["n_trades"],
            "PnL_USDT": round(res["total_pnl_usdt"], 1),
            "maxDD_USDT": round(res["max_drawdown_usdt"], 1),
            "Sharpe": round(res["sharpe"], 2),
            "PnL/DD": (round(res["total_pnl_usdt"] / abs(res["max_drawdown_usdt"]), 2)
                       if res["max_drawdown_usdt"] else float("inf")),
        })
        print(f"  → пар={res['pairs_traded']} сделок={res['n_trades']} "
              f"PnL={res['total_pnl_usdt']:+.0f} DD={res['max_drawdown_usdt']:.0f} "
              f"Sharpe={res['sharpe']:.2f}\n")

    print("=" * 72)
    print("СРАВНЕНИЕ ТАЙМФРЕЙМОВ (одинаковое окно, out-of-sample)")
    print("=" * 72)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nPnL/DD — прибыль на единицу просадки (чем больше, тем лучше риск/доход).")
    print("Sharpe — годовая доходность на единицу риска.")


if __name__ == "__main__":
    main()
