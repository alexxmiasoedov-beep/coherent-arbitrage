#!/usr/bin/env python3
"""Скан юниверса на коинтеграцию для заданного таймфрейма (с кэшем данных).

Показывает ВСЕ пары с p-value < COINT_PVALUE_MAX, их half-life и текущий z-score.

    python scan_cointegration.py [interval] [bars] [top_n]
    python scan_cointegration.py 1m 14400 150

Скан двухфазный:
  1) БЫСТРЫЙ скрининг всех пар параллельно на всех ядрах (coint с одним лагом).
  2) ТОЧНЫЙ ретест выживших (p<порог) с автоподбором лага (autolag='aic').
Это на порядок быстрее наивного однопоточного coint по длинным рядам.
"""
from __future__ import annotations

# Пиннинг BLAS до импорта numpy/pandas: иначе каждый coint плодит потоки и
# параллельные воркеры дерутся за ядра. 1 поток на процесс → чистое N-кратное ускорение.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from itertools import combinations
from multiprocessing import Pool

import pandas as pd

import config
from src import data
from src.cointegration import analyze_pair

# Кэш скачанных цен: репо-локальный .cache/ (в .gitignore). Можно переопределить
# через переменную окружения ARB_CACHE_DIR.
CACHE_DIR = os.environ.get(
    "ARB_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache"),
)
os.makedirs(CACHE_DIR, exist_ok=True)

N_WORKERS = max(1, (os.cpu_count() or 2))

_PRICES: pd.DataFrame | None = None


def _init_worker(prices: pd.DataFrame) -> None:
    global _PRICES
    _PRICES = prices


def _screen_pair(pair: tuple[str, str]) -> dict | None:
    """Фаза 1: быстрый скрининг одной пары (coint с maxlag=1, без autolag)."""
    y, x = pair
    try:
        r = analyze_pair(_PRICES[y], _PRICES[x], coint_maxlag=1, coint_autolag=None)
    except Exception:  # noqa: BLE001
        return None
    if r["pvalue"] < config.COINT_PVALUE_MAX:
        return {"y": y, "x": x, "pvalue": r["pvalue"],
                "half_life_bars": r["half_life_h"], "current_z": r["current_z"]}
    return None


def load_prices(interval: str, n_bars: int, top_n: int) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"px_{interval}_{n_bars}_{top_n}.pkl")
    if os.path.exists(cache):
        print(f"Кэш найден: {cache}")
        return pd.read_pickle(cache)
    symbols = data.top_symbols(top_n, use_whitelist=False)
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
    print(f"Фаза 1: быстрый скрининг {len(pairs)} пар на {N_WORKERS} ядрах...", flush=True)

    with Pool(N_WORKERS, initializer=_init_worker, initargs=(prices,)) as pool:
        screened = [r for r in pool.imap_unordered(_screen_pair, pairs, chunksize=16) if r]

    print(f"Прошли скрининг (p<{config.COINT_PVALUE_MAX}): {len(screened)}. "
          f"Фаза 2: точный ретест (autolag='aic')...", flush=True)

    rows = []
    for cand in screened:
        try:
            r = analyze_pair(prices[cand["y"]], prices[cand["x"]])  # autolag='aic'
        except Exception:  # noqa: BLE001
            continue
        if r["pvalue"] < config.COINT_PVALUE_MAX:
            rows.append({"y": cand["y"], "x": cand["x"], "pvalue": r["pvalue"],
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
    out = os.path.join(CACHE_DIR, f"pairs_{interval}_{n_bars}_{top_n}.csv")
    df.to_csv(out, index=False)
    print(df.head(40).to_string(index=False))
    print(f"\nВсего сохранено в {out}")
    print("half_life_bars — полураспад в БАРАХ (на 1м = минуты); current_z — текущее отклонение.")


if __name__ == "__main__":
    main()
