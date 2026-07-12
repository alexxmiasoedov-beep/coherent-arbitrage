#!/usr/bin/env python3
"""Глубокий разбор конкретной пары: полный coint-тест, график, мини-бэктест.

    python deep_dive.py SUI-USDT ENS-USDT [cache_pkl]

Отличие от скрининга: p-value считается ПОЛНЫМ тестом (autolag='aic'), а не
быстрым (maxlag=1). Строит график спреда+z-score (PNG) и гоняет наивную
mean-reversion стратегию out-of-sample (форма 60% / тест 40%).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

import config
from src.cointegration import hedge_ratio, spread, half_life

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, ".cache", "px_1m_14400_80.pkl")
OUTDIR = os.path.join(HERE, ".cache")


def backtest(s: pd.Series, mean: float, std: float) -> dict:
    """Наивный mean-reversion по z-score спреда. PnL в единицах спреда.

    Вход при |z|>=ENTRY_Z, выход при |z|<=EXIT_Z, стоп при |z|>=STOP_Z.
    Позиция по спреду: -1 (шорт спред) при z>0, +1 (лонг спред) при z<0.
    """
    z = (s - mean) / std
    pos = 0            # -1 шорт спред, +1 лонг, 0 плоско
    entry_s = 0.0
    trades = []
    for t in range(len(s)):
        zt = z.iloc[t]
        st = s.iloc[t]
        if pos == 0:
            if zt >= config.ENTRY_Z:
                pos, entry_s = -1, st
            elif zt <= -config.ENTRY_Z:
                pos, entry_s = 1, st
        else:
            hit_tp = abs(zt) <= config.EXIT_Z
            hit_sl = abs(zt) >= config.STOP_Z
            if hit_tp or hit_sl:
                pnl = pos * (st - entry_s)   # шорт спред зарабатывает на падении s
                trades.append({"pnl": pnl, "stop": hit_sl})
                pos = 0
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        "n_trades": len(trades),
        "n_stops": sum(t["stop"] for t in trades),
        "total_pnl_spread": float(np.sum(pnls)) if pnls else 0.0,
        "avg_sigma": float(np.mean(pnls) / std) if pnls else 0.0,
        "win_rate": len(wins) / len(pnls) if pnls else 0.0,
    }


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: python deep_dive.py Y-USDT X-USDT [cache_pkl]")
        sys.exit(1)
    y_sym, x_sym = sys.argv[1], sys.argv[2]
    cache = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CACHE
    px = pd.read_pickle(cache)
    y, x = px[y_sym], px[x_sym]
    n = len(y)

    # Полный тест на всей выборке
    _, pval_full, _ = coint(y.values, x.values)     # autolag='aic' по умолчанию
    _, pval_fast, _ = coint(y.values, x.values, maxlag=1, autolag=None)
    beta = hedge_ratio(y, x)
    s_all = spread(y, x, beta)
    hl = half_life(s_all)

    # In-sample форма (60%) / out-of-sample тест (40%)
    cut = int(n * 0.6)
    s_form, s_test = s_all.iloc[:cut], s_all.iloc[cut:]
    mean, std = s_form.mean(), s_form.std()
    bt = backtest(s_test, mean, std)

    print(f"=== {y_sym} / {x_sym} ===")
    print(f"баров: {n}  |  beta: {beta:.4f}  |  half-life: {hl:.0f} мин")
    print(f"p-value: полный тест {pval_full:.4f}  |  быстрый (скрининг) {pval_fast:.4f}")
    print(f"спред форма: mean={mean:.4f} std={std:.4f}")
    print(f"\nOut-of-sample бэктест (тест-окно {len(s_test)} баров ≈ "
          f"{len(s_test)/60:.1f} ч):")
    print(f"  сделок: {bt['n_trades']}  (из них по стопу: {bt['n_stops']})")
    print(f"  win-rate: {bt['win_rate']*100:.0f}%")
    print(f"  суммарный PnL: {bt['total_pnl_spread']:.4f} ед. спреда "
          f"= {bt['total_pnl_spread']/std:.2f}σ")
    print(f"  ⚠ без комиссий/проскальзывания; naive-стратегия (см. README: нет эджа)")

    # График: z-score с порогами
    z_all = (s_all - mean) / std
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(z_all.index, z_all.values, lw=0.6, color="#2563eb", label="z-score спреда")
    for lvl, c, ls in [(config.ENTRY_Z, "#dc2626", "--"), (-config.ENTRY_Z, "#dc2626", "--"),
                       (config.EXIT_Z, "#16a34a", ":"), (-config.EXIT_Z, "#16a34a", ":"),
                       (config.STOP_Z, "#000", "-."), (-config.STOP_Z, "#000", "-.")]:
        ax.axhline(lvl, color=c, ls=ls, lw=0.8)
    ax.axvline(z_all.index[cut], color="#888", lw=1.2, label="форма|тест split")
    ax.set_title(f"{y_sym} / {x_sym}  —  z-score спреда (1m)  "
                 f"β={beta:.3f}, half-life={hl:.0f}м, p={pval_full:.3f}")
    ax.set_ylabel("z-score (σ)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    png = os.path.join(OUTDIR, f"spread_{y_sym.split('-')[0]}_{x_sym.split('-')[0]}.png")
    fig.savefig(png, dpi=110)
    print(f"\nГрафик: {png}")


if __name__ == "__main__":
    main()
