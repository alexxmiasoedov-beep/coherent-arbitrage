#!/usr/bin/env python3
"""Скан юниверса на коинтеграцию для заданного таймфрейма (с кэшем данных).

Показывает ВСЕ пары с p-value < COINT_PVALUE_MAX, их half-life и текущий z-score.

    python scan_cointegration.py [interval] [bars] [top_n]
    python scan_cointegration.py 1m 14400 20
"""
from __future__ import annotations

import os
import sys
from itertools import combinations

import pandas as pd

import config
from src import data
from src.cointegration import analyze_pair

CACHE_DIR = "/tmp/claude-1000/-home-miasoedov-projects--/05309a0a-20ed-47bd-bc17-eb4991e34f0b/scratchpad"


def load_prices(interval: str, n_bars: int, top_n: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"px_{interval}_{n_bars}_{top_n}.pkl")
    if os.path.exists(cache):
        print(f"Кэш найден: {cache}")
        return pd.read_pickle(cache)
    symbols = data.top_symbols(top_n)
    print(f"Гружу {interval} {n_bars} баров по {len(symbols)} контрактам (пагинация)...")
    prices = data.price_matrix_n(symbols, interval, n_bars)
    prices.to_pickle(cache)
    print(f"Сохранил кэш: {cache}")
    return prices


def main() -> None:
    interval = sys.argv[1] if len(sys.argv) > 1 else "1m"
    n_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 14400
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    prices = load_prices(interval, n_bars, top_n)
    print(f"История: {prices.shape[0]} баров × {prices.shape[1]} контрактов")
    pairs = list(combinations(prices.columns, 2))
    print(f"Проверяю {len(pairs)} пар на коинтеграцию...", flush=True)

    rows = []
    for y, x in pairs:
        try:
            r = analyze_pair(prices[y], prices[x])
        except Exception:  # noqa: BLE001
            continue
        if r["pvalue"] < config.COINT_PVALUE_MAX:
            rows.append({"y": y, "x": x, "pvalue": r["pvalue"],
                         "half_life_bars": r["half_life_h"], "current_z": r["current_z"]})

    df = pd.DataFrame(rows)
    print(f"\n=== Коинтегрированных пар (p<{config.COINT_PVALUE_MAX}): "
          f"{len(df)} из {len(pairs)} ({(len(df)/len(pairs)*100 if pairs else 0):.1f}%) ===\n", flush=True)
    if df.empty:
        print("Ничего не найдено.")
        return
    df = df.sort_values("pvalue").reset_index(drop=True)
    df["pvalue"] = df["pvalue"].round(4)
    df["half_life_bars"] = df["half_life_bars"].round(0)
    df["current_z"] = df["current_z"].round(2)
    print(df.head(40).to_string(index=False))
    print("\nhalf_life_bars — полураспад в БАРАХ (на 1м = минуты); current_z — текущее отклонение.")


if __name__ == "__main__":
    main()
