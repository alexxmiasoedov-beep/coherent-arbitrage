#!/usr/bin/env python3
"""Перебор параметров стратегии на одних и тех же данных — ищем, есть ли плюс.

    python run_sweep.py
"""
from __future__ import annotations

import pandas as pd

from src import data
from src.backtest import run, Params

CONFIGS = {
    "baseline (entry2.0)":            Params(entry_z=2.0),
    "entry2.5":                       Params(entry_z=2.5),
    "entry3.0":                       Params(entry_z=3.0),
    "entry2.5 +cooldown3":            Params(entry_z=2.5, cooldown_bars=3),
    "entry2.5 +cost_gate":            Params(entry_z=2.5, cost_gate=True),
    "entry2.5 +timestop3hl":          Params(entry_z=2.5, max_hold_mult=3.0),
    "entry2.5 +all":                  Params(entry_z=2.5, cooldown_bars=3,
                                             max_hold_mult=3.0, cost_gate=True),
    "entry3.0 +all":                  Params(entry_z=3.0, cooldown_bars=5,
                                             max_hold_mult=3.0, cost_gate=True),
    "entry2.5 +all, maker fee":       Params(entry_z=2.5, cooldown_bars=3,
                                             max_hold_mult=3.0, cost_gate=True, fee=0.0002),
}


def main() -> None:
    symbols = data.top_symbols()
    print(f"Гружу историю ({len(symbols)} контрактов)...")
    prices = data.price_matrix(symbols)
    print(f"История: {prices.shape}. Прогоняю {len(CONFIGS)} конфигураций...\n")

    rows = []
    for name, p in CONFIGS.items():
        res = run(prices=prices, params=p, verbose=False)
        rows.append({
            "config": name,
            "pairs": res["pairs_traded"],
            "trades": res["n_trades"],
            "PnL": round(res["total_pnl_usdt"], 1),
            "maxDD": round(res["max_drawdown_usdt"], 1),
            "Sharpe": round(res["sharpe"], 2),
            "PnL/DD": (round(res["total_pnl_usdt"] / abs(res["max_drawdown_usdt"]), 2)
                       if res["max_drawdown_usdt"] else 0.0),
        })
        print(f"  {name:32s} PnL={res['total_pnl_usdt']:+7.1f} "
              f"DD={res['max_drawdown_usdt']:7.1f} Sharpe={res['sharpe']:+.2f} "
              f"trades={res['n_trades']}")

    print("\n" + "=" * 78)
    print("ПЕРЕБОР ПАРАМЕТРОВ (30 дней, out-of-sample, чистая крипта)")
    print("=" * 78)
    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print(df.to_string(index=False))
    best = df.iloc[0]
    print(f"\nЛучшая по Sharpe: {best['config']} "
          f"(PnL={best['PnL']}, Sharpe={best['Sharpe']}, сделок={best['trades']})")
    if best["PnL"] <= 0:
        print("⚠️  Даже лучшая конфигурация убыточна — простого преимущества здесь нет.")


if __name__ == "__main__":
    main()
