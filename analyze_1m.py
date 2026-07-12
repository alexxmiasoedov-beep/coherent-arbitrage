#!/usr/bin/env python3
"""Полный отбор пар на 1m из кэша цен по НАШИМ условиям.

Условия (config.py):
  - коинтеграция: p-value < COINT_PVALUE_MAX (0.05)
  - half-life спреда в [MIN_HALFLIFE_H, MAX_HALFLIFE_H] баров (на 1m = минуты)
  - помечаем готовые ко входу: |current_z| >= ENTRY_Z (2.0)

Читает готовый кэш .cache/px_1m_14400_80.pkl (не качает заново).
Пишет прогресс каждые PROGRESS пар и итог в .cache/pairs_1m.csv.

    python analyze_1m.py [limit_pairs]   # limit — для быстрого замера скорости
"""
from __future__ import annotations

import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import config
from src.cointegration import hedge_ratio, spread, half_life, zscore

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     ".cache", "px_1m_14400_80.pkl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   ".cache", "pairs_1m.csv")
PROGRESS = 100


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    prices = pd.read_pickle(CACHE)
    print(f"История: {prices.shape[0]} баров × {prices.shape[1]} контрактов",
          flush=True)

    pairs = list(combinations(prices.columns, 2))
    if limit:
        pairs = pairs[:limit]
    print(f"Проверяю {len(pairs)} пар...", flush=True)

    rows = []
    t0 = time.time()
    for i, (y, x) in enumerate(pairs, 1):
        sy, sx = prices[y], prices[x]
        try:
            # Быстрый тест: фиксированный лаг, без autolag (в ~50× быстрее).
            _, pvalue, _ = coint(sy.values, sx.values, maxlag=1, autolag=None)
        except Exception:  # noqa: BLE001
            continue
        if pvalue < config.COINT_PVALUE_MAX:
            # Дорогие метрики считаем только для прошедших порог.
            beta = hedge_ratio(sy, sx)
            s = spread(sy, sx, beta)
            z = zscore(s, 0).dropna()
            rows.append({
                "y": y, "x": x,
                "pvalue": float(pvalue),
                "half_life": half_life(s),   # на 1m = минуты
                "beta": beta,
                "current_z": float(z.iloc[-1]) if len(z) else np.nan,
            })
        if i % PROGRESS == 0:
            dt = time.time() - t0
            rate = i / dt
            eta = (len(pairs) - i) / rate
            print(f"  {i}/{len(pairs)} пар | {len(rows)} коинтегр. | "
                  f"{rate:.1f} пар/с | ETA {eta:.0f}с", flush=True)

    df = pd.DataFrame(rows)
    total = len(pairs)
    coint_n = len(df)
    print(f"\n=== Коинтегрированных (p<{config.COINT_PVALUE_MAX}): "
          f"{coint_n} из {total} ({coint_n/total*100:.1f}%) ===", flush=True)
    if df.empty:
        print("Ничего не найдено.")
        return

    # Полные условия: + half-life в диапазоне
    hl_lo, hl_hi = config.MIN_HALFLIFE_H, config.MAX_HALFLIFE_H
    df["hl_ok"] = df["half_life"].between(hl_lo, hl_hi)
    df["entry_ready"] = df["current_z"].abs() >= config.ENTRY_Z

    passing = df[df["hl_ok"]].copy()
    print(f"    из них half-life в [{hl_lo},{hl_hi}] баров: {len(passing)}",
          flush=True)
    print(f"    из них готовы ко входу |z|>={config.ENTRY_Z}: "
          f"{int(df['entry_ready'].sum())} (среди всех коинтегр.)", flush=True)

    df = df.sort_values("pvalue").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"\nСохранил все коинтегр. пары: {OUT}", flush=True)

    show = df.copy()
    show["pvalue"] = show["pvalue"].round(4)
    show["half_life"] = show["half_life"].round(0)
    show["beta"] = show["beta"].round(4)
    show["current_z"] = show["current_z"].round(2)
    cols = ["y", "x", "pvalue", "half_life", "beta", "current_z",
            "hl_ok", "entry_ready"]
    print("\n=== ТОП-50 по p-value ===", flush=True)
    print(show[cols].head(50).to_string(index=False), flush=True)
    print("\nhalf_life — полураспад в БАРАХ (1m = минуты). "
          "hl_ok — прошёл фильтр half-life. entry_ready — |z|>=2 сейчас.",
          flush=True)


if __name__ == "__main__":
    main()
