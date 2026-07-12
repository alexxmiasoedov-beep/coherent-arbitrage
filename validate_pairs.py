#!/usr/bin/env python3
"""Walk-forward валидация КОНКРЕТНЫХ пар-кандидатов из скана (out-of-sample).

Скан находит пары, коинтегрированные на ВСЁМ окне — это in-sample подгонка.
Здесь каждую пару гоняем walk-forward'ом: несколько независимых фолдов, в каждом
formation-часть оценивает β/μ/σ/half-life и ПЕРЕПРОВЕРЯЕТ коинтеграцию, а на test-части
торгуем зафиксированной конфигурацией. Оставляем пары, что держат плюс на РАЗНЫХ
периодах — это отсекает переобучение.

    python validate_pairs.py [pairs_csv] [prices_pkl]
    python validate_pairs.py .cache/pairs_1m_14400_80.csv .cache/px_1m_14400_80.pkl
"""
from __future__ import annotations

# Пиннинг BLAS до numpy — чтобы параллельные воркеры не дрались за ядра.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from multiprocessing import Pool

import pandas as pd

from src.backtest import Params, _simulate_pair

# --- Схема walk-forward (в барах; на 1m = минуты) ---
FORM = 2880    # formation на фолд (~2 дня 1m) — оценка β/μ/σ/half-life + coint-тест
TEST = 1440    # test (OOS) на фолд (~1 день)
STEP = 1440    # сдвиг между фолдами (непересекающиеся test-окна)

# Та же «победная» конфигурация, что валидировали в walk_forward.py.
PARAMS = Params(entry_z=3.0, exit_z=0.5, stop_z=3.5,
                cooldown_bars=5, max_hold_mult=3.0, cost_gate=True)

# Критерии «робастной» пары (переживает переобучение):
MIN_TRADED_FOLDS = 4       # торговала (была коинтегрирована) хотя бы в стольких фолдах
MIN_POS_SHARE = 0.60       # доля прибыльных фолдов среди торговавших
MIN_TOTAL_PNL = 0.0        # суммарный OOS PnL положительный

_PRICES: pd.DataFrame | None = None


def _init(prices: pd.DataFrame) -> None:
    global _PRICES
    _PRICES = prices


def _validate_pair(pair: tuple[str, str]) -> dict:
    y, x = pair
    ya, xa = _PRICES[y], _PRICES[x]
    n = len(_PRICES)
    traded = 0
    pos = 0
    total_pnl = 0.0
    total_trades = 0
    wins = 0.0
    start = 0
    n_folds = 0
    while start + FORM + TEST <= n:
        n_folds += 1
        yf, xf = ya.iloc[start:start + FORM], xa.iloc[start:start + FORM]
        yt = ya.iloc[start + FORM:start + FORM + TEST]
        xt = xa.iloc[start + FORM:start + FORM + TEST]
        try:
            r = _simulate_pair(yf, xf, yt, xt, PARAMS)
        except Exception:  # noqa: BLE001
            r = None
        if r:
            traded += 1
            total_pnl += r["net_pnl"]
            total_trades += r["n_trades"]
            wins += r["win_rate"]
            if r["net_pnl"] > 0:
                pos += 1
        start += STEP
    return {
        "y": y, "x": x, "folds": n_folds, "traded_folds": traded,
        "pos_folds": pos,
        "pos_share": round(pos / traded, 2) if traded else 0.0,
        "total_pnl": round(total_pnl, 2),
        "trades": total_trades,
        "avg_win_rate": round(wins / traded, 2) if traded else 0.0,
    }


def main() -> None:
    pairs_csv = sys.argv[1] if len(sys.argv) > 1 else ".cache/pairs_1m_14400_80.csv"
    prices_pkl = sys.argv[2] if len(sys.argv) > 2 else ".cache/px_1m_14400_80.pkl"

    prices = pd.read_pickle(prices_pkl)
    cand = pd.read_csv(pairs_csv)
    pairs = list(zip(cand["y"], cand["x"]))
    n_workers = max(1, (os.cpu_count() or 2))
    n_folds_est = 1 + (len(prices) - FORM - TEST) // STEP
    print(f"Данные: {prices.shape[0]} баров × {prices.shape[1]} контрактов")
    print(f"Кандидатов: {len(pairs)} | walk-forward: FORM={FORM} TEST={TEST} STEP={STEP} "
          f"→ ~{n_folds_est} фолдов/пару | конфиг: entry_z={PARAMS.entry_z} cost_gate=on")
    print(f"Гоняю на {n_workers} ядрах...", flush=True)

    with Pool(n_workers, initializer=_init, initargs=(prices,)) as pool:
        rows = list(pool.imap_unordered(_validate_pair, pairs, chunksize=8))

    df = pd.DataFrame(rows)
    robust = df[(df["traded_folds"] >= MIN_TRADED_FOLDS)
                & (df["pos_share"] >= MIN_POS_SHARE)
                & (df["total_pnl"] > MIN_TOTAL_PNL)].copy()
    robust = robust.sort_values(["pos_share", "total_pnl"], ascending=False).reset_index(drop=True)

    out = pairs_csv.replace("pairs_", "validated_")
    robust.to_csv(out, index=False)

    print(f"\n{'='*72}")
    print(f"WALK-FORWARD ВАЛИДАЦИЯ: {len(pairs)} кандидатов → {len(robust)} робастных")
    print(f"(критерий: торговала ≥{MIN_TRADED_FOLDS} фолдов, ≥{int(MIN_POS_SHARE*100)}% "
          f"фолдов в плюс, суммарный OOS PnL > 0)")
    print(f"{'='*72}\n")

    if robust.empty:
        print("🔴 Ни одна пара не прошла — коинтеграция на скане была подгонкой,")
        print("   OOS-плюса на разных периодах нет. (Совпадает с прошлым выводом.)")
    else:
        print(robust.head(40).to_string(index=False))
        print(f"\nСохранено в {out}")
        # общая картина по всем торговавшим
        tr = df[df["traded_folds"] > 0]
        print(f"\nПо всем кандидатам: суммарный OOS PnL={df['total_pnl'].sum():+.1f} USDT | "
              f"пар с плюсом={int((df['total_pnl']>0).sum())}/{len(df)} | "
              f"медианный pos_share={tr['pos_share'].median():.2f}")


if __name__ == "__main__":
    main()
