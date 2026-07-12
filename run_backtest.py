#!/usr/bin/env python3
"""Бэктест стратегии на исторических данных BingX.

    python run_backtest.py
"""
from __future__ import annotations

import pandas as pd

import config
from src import data
from src.backtest import run


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    symbols = data.top_symbols()
    res = run(symbols)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ БЭКТЕСТА (out-of-sample)")
    print("=" * 60)
    print(f"Пар торговалось:        {res['pairs_traded']}")
    print(f"Всего сделок:           {res['n_trades']}")
    print(f"Итоговый PnL:           {res['total_pnl_usdt']:+.2f} USDT")
    print(f"Макс. просадка:         {res['max_drawdown_usdt']:+.2f} USDT")
    print(f"Sharpe (годовой):       {res['sharpe']:.2f}")
    print(f"Доля прибыльных пар:    {res['profitable_pairs_share']*100:.0f}%")
    print(f"Капитал на ногу:        {config.CAPITAL_PER_LEG_USDT} USDT | "
          f"комиссия тейкера {config.TAKER_FEE*100:.3f}%")

    if not res["table"].empty:
        t = res["table"][["y", "x", "pvalue", "half_life_h", "n_trades", "net_pnl", "win_rate"]].copy()
        t["pvalue"] = t["pvalue"].round(4)
        t["half_life_h"] = t["half_life_h"].round(1)
        t["net_pnl"] = t["net_pnl"].round(2)
        t["win_rate"] = (t["win_rate"] * 100).round(0)
        print("\nТоп пар по PnL:")
        print(t.head(15).to_string(index=False))
        print("\n(win_rate — доля прибыльных сделок внутри пары, %)")
    else:
        print("\nНи одна пара не дала сделок на тестовом окне.")


if __name__ == "__main__":
    main()
