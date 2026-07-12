"""Математика статистического арбитража: коинтеграция, спред, z-score, half-life.

Модель пары:
    spread_t = P_y,t - beta * P_x,t   (beta — коэффициент хеджа, "гамма")
Если spread стационарен (возвращается к среднему), пара коинтегрирована.
Сигнал = z-score спреда: насколько текущий спред отклонился от своего среднего.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS-регрессия y на x (с константой). Возвращает beta (наклон) — коэффициент хеджа."""
    X = sm.add_constant(x.values)
    model = sm.OLS(y.values, X).fit()
    return float(model.params[1])


def spread(y: pd.Series, x: pd.Series, beta: float) -> pd.Series:
    """Спред пары при заданном beta."""
    return y - beta * x


def zscore(s: pd.Series, window: int = 0) -> pd.Series:
    """Z-score спреда. window=0 → по всей выборке; иначе — скользящее окно."""
    if window and window > 0:
        mean = s.rolling(window).mean()
        std = s.rolling(window).std()
    else:
        mean = s.mean()
        std = s.std()
    return (s - mean) / std


def half_life(s: pd.Series) -> float:
    """Период полураспада спреда (Ornstein-Uhlenbeck).

    Регрессия Δspread_t = a + lambda * spread_{t-1}.
    half-life = -ln(2) / lambda. Чем меньше — тем быстрее возврат к среднему.
    Возвращает inf, если возврата нет (lambda >= 0).
    """
    s = s.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    X = sm.add_constant(lag.values)
    lam = float(sm.OLS(delta.values, X).fit().params[1])
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def engle_granger_pvalue(y: pd.Series, x: pd.Series) -> float:
    """P-value теста Энгла-Грейнджера на коинтеграцию. Меньше → сильнее коинтеграция."""
    _, pvalue, _ = coint(y.values, x.values)
    return float(pvalue)


def analyze_pair(y: pd.Series, x: pd.Series, z_window: int = 0) -> dict:
    """Полный разбор пары (y, x): p-value, beta, half-life, среднее/σ спреда, текущий z.

    spread_mean и spread_std сохраняются, чтобы бот считал live z-score из текущих цен:
        z_now = (P_y - beta*P_x - spread_mean) / spread_std
    """
    beta = hedge_ratio(y, x)
    s = spread(y, x, beta)
    z = zscore(s, z_window)
    current_z = float(z.dropna().iloc[-1]) if len(z.dropna()) else float("nan")
    return {
        "pvalue": engle_granger_pvalue(y, x),
        "beta": beta,
        "half_life_h": half_life(s),
        "spread_mean": float(s.mean()),
        "spread_std": float(s.std()),
        "current_z": current_z,
    }


def live_zscore(price_y: float, price_x: float, beta: float,
                spread_mean: float, spread_std: float) -> float:
    """Z-score спреда по текущим ценам двух ног и сохранённым mean/std."""
    if spread_std == 0:
        return float("nan")
    return (price_y - beta * price_x - spread_mean) / spread_std
