#!/usr/bin/env python3
"""Точка входа: находит коинтегрированные крипто-пары и показывает сигналы.

Запуск:
    python run_screener.py
"""
from __future__ import annotations

import pandas as pd

import config
from src import data
from src.screener import screen, label_signal


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    symbols = data.top_symbols()
    result = screen(symbols)

    if result.empty:
        print("\nКоинтегрированных пар не найдено при текущих порогах.")
        print("Попробуй ослабить COINT_PVALUE_MAX или расширить окно LOOKBACK в config.py.")
        return

    result["signal"] = result["current_z"].apply(label_signal)

    show = result[["y", "x", "pvalue", "beta", "half_life_h", "current_z", "signal"]].copy()
    show["pvalue"] = show["pvalue"].round(4)
    show["beta"] = show["beta"].round(4)
    show["half_life_h"] = show["half_life_h"].round(1)
    show["current_z"] = show["current_z"].round(2)

    print(f"\n=== Найдено {len(show)} коинтегрированных пар "
          f"(p<{config.COINT_PVALUE_MAX}, half-life {config.MIN_HALFLIFE_H}-{config.MAX_HALFLIFE_H}ч) ===\n")
    print(show.to_string(index=False))

    actionable = show[show["signal"].str.startswith("ВХОД")]
    if not actionable.empty:
        print(f"\n>>> Пары с сигналом на вход прямо сейчас: {len(actionable)}")
    else:
        print("\n>>> Сигналов на вход сейчас нет (все около среднего). Это норма — ждём отклонения.")


if __name__ == "__main__":
    main()
