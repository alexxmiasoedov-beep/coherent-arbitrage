"""Скринер: перебирает все пары из вселенной и отбирает коинтегрированные."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

import config
from src import data
from src.cointegration import analyze_pair


def screen(symbols: list[str] | None = None, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Прогоняет все сочетания пар и возвращает отсортированную таблицу кандидатов.

    Отбор: p-value < COINT_PVALUE_MAX и half-life в диапазоне [MIN, MAX].
    Сортировка: по p-value (сильнее коинтеграция — выше), затем по |current_z|
    (больше отклонение сейчас — интереснее для входа).
    """
    if prices is None:
        symbols = symbols or data.top_symbols()
        print(f"Загружаю цены по {len(symbols)} монетам ({config.INTERVAL}, "
              f"{config.LOOKBACK} свечей)...")
        prices = data.price_matrix(symbols)
    symbols = list(prices.columns)
    print(f"Готово: {len(prices)} общих точек по {len(symbols)} монетам.")

    pairs = list(combinations(symbols, 2))
    print(f"Проверяю {len(pairs)} пар на коинтеграцию...")

    rows = []
    for y_sym, x_sym in pairs:
        try:
            res = analyze_pair(prices[y_sym], prices[x_sym], config.ZSCORE_WINDOW)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {y_sym}/{x_sym}: {e}")
            continue
        rows.append({"y": y_sym, "x": x_sym, **res})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    keep = (
        (df["pvalue"] < config.COINT_PVALUE_MAX)
        & (df["half_life_h"] >= config.MIN_HALFLIFE_H)
        & (df["half_life_h"] <= config.MAX_HALFLIFE_H)
    )
    out = df[keep].copy()
    out["abs_z"] = out["current_z"].abs()
    out = out.sort_values(["pvalue", "abs_z"], ascending=[True, False]).reset_index(drop=True)
    return out


def label_signal(z: float) -> str:
    """Текстовый сигнал по текущему z-score и порогам из config."""
    az = abs(z)
    if az > config.STOP_Z:
        return "СТОП (коинтеграция под вопросом)"
    if az > config.ENTRY_Z:
        side = "LONG spread" if z < 0 else "SHORT spread"
        return f"ВХОД: {side}"
    if az <= config.EXIT_Z:
        return "около среднего (ждать)"
    return "наблюдать"
