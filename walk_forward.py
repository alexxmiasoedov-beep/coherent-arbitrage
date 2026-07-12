#!/usr/bin/env python3
"""Walk-forward валидация: тот же приём, но на НЕСКОЛЬКИХ независимых окнах.

Берём длинную историю и скользим окном: каждый фолд оценивает параметры на своей
formation-части и торгует на своей test-части. Если фиксированная конфигурация
прибыльна на РАЗНЫХ периодах — есть преимущество. Если скачет между плюсом и
минусом — это была подгонка под одно окно.

    python walk_forward.py
"""
from __future__ import annotations

import pandas as pd

from src import data
from src.backtest import run, Params

TOTAL_BARS = 1440      # ~60 дней 1h
FORM = 480             # formation на фолд
TEST = 240             # test на фолд
STEP = 240             # сдвиг между фолдами

# Конфигурация-победитель из перебора — проверяем именно её честность.
PARAMS = Params(entry_z=3.0, cooldown_bars=5, max_hold_mult=3.0, cost_gate=True)


def main() -> None:
    symbols = data.top_symbols()
    print(f"Гружу {TOTAL_BARS} баров 1h по {len(symbols)} контрактам (пагинация)...")
    prices = data.price_matrix_n(symbols, "1h", TOTAL_BARS)
    print(f"История: {prices.shape} | {prices.index[0]} -> {prices.index[-1]}\n")

    rows = []
    start = 0
    fold = 0
    while start + FORM + TEST <= len(prices):
        sl = prices.iloc[start:start + FORM + TEST]
        res = run(prices=sl, form_bars=FORM, params=PARAMS, verbose=False)
        test_start = sl.index[FORM]
        rows.append({
            "fold": fold,
            "test_from": str(test_start.date()),
            "pairs": res["pairs_traded"],
            "trades": res["n_trades"],
            "PnL": round(res["total_pnl_usdt"], 1),
            "maxDD": round(res["max_drawdown_usdt"], 1),
            "Sharpe": round(res["sharpe"], 2),
        })
        print(f"  fold {fold} (test с {test_start.date()}): "
              f"PnL={res['total_pnl_usdt']:+7.1f} Sharpe={res['sharpe']:+.2f} "
              f"trades={res['n_trades']}")
        start += STEP
        fold += 1

    print("\n" + "=" * 66)
    print("WALK-FORWARD: конфигурация entry3.0 +all на разных периодах")
    print("=" * 66)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    pos = (df["PnL"] > 0).sum()
    total_pnl = df["PnL"].sum()
    print(f"\nПрибыльных фолдов: {pos} из {len(df)} | суммарный PnL: {total_pnl:+.1f} USDT")
    if pos == len(df):
        print("✅ Плюс на ВСЕХ периодах — есть признак реального преимущества.")
    elif pos >= len(df) * 0.6:
        print("🟡 Плюс на большинстве, но не везде — преимущество слабое/нестабильное.")
    else:
        print("🔴 Плюс не держится на разных периодах — это была подгонка, а не эдж.")


if __name__ == "__main__":
    main()
