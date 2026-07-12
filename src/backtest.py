"""Бэктест стратегии на истории с честным out-of-sample.

Чтобы не переобучиться, окно делится на две части:
  • FORMATION (первые BACKTEST_FORM_BARS свечей): здесь оцениваем β, среднее и σ
    спреда и проверяем коинтеграцию. Торговых решений тут НЕТ.
  • TEST (остаток): торгуем по сигналу z-score, используя ЗАФИКСИРОВАННЫЕ на
    formation параметры. Так мы измеряем стратегию на данных, которых она «не видела».

PnL считается той же логикой, что и в живом боте (номинал на ногу, комиссии тейкера).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import data
from src.cointegration import hedge_ratio, spread, half_life, engle_granger_pvalue


def _simulate_pair(y_form, x_form, y_test, x_test) -> dict | None:
    """Один бэктест пары. Возвращает метрики и ряд PnL по барам теста, либо None."""
    # --- параметры фиксируем на formation ---
    pval = engle_granger_pvalue(y_form, x_form)
    if pval >= config.COINT_PVALUE_MAX:
        return None
    beta = hedge_ratio(y_form, x_form)
    s_form = spread(y_form, x_form, beta)
    mu, sigma = float(s_form.mean()), float(s_form.std())
    hl = half_life(s_form)
    if sigma == 0 or not (config.MIN_HALFLIFE_H <= hl <= config.MAX_HALFLIFE_H):
        return None

    # --- z-score на тесте по зафиксированным mu/sigma ---
    s_test = y_test - beta * x_test
    z = (s_test - mu) / sigma

    # --- пошаговая симуляция входов/выходов ---
    position = 0            # 0 flat, +1 long spread, -1 short spread
    entry_py = entry_px = 0.0
    qty_y = qty_x = 0.0
    bar_pnl = pd.Series(0.0, index=z.index)
    trades = []             # реализованный PnL по каждой сделке

    py = y_test.values
    px = x_test.values
    zz = z.values
    idx = z.index

    def leg_fee(price, qty):
        return price * qty * config.TAKER_FEE

    prev_equity_y = prev_equity_x = 0.0
    for i in range(len(zz)):
        zi = zz[i]
        # --- управление открытой позицией ---
        if position != 0:
            # плавающий PnL с прошлого бара → в bar_pnl (для equity/Sharpe)
            upnl = (position * qty_y * (py[i] - entry_py)
                    - position * qty_x * (px[i] - entry_px))
            bar_pnl.iloc[i] = upnl - (prev_equity_y + prev_equity_x)
            prev_equity_y = position * qty_y * (py[i] - entry_py)
            prev_equity_x = -position * qty_x * (px[i] - entry_px)

            if abs(zi) <= config.EXIT_Z or abs(zi) >= config.STOP_Z:
                gross = (position * qty_y * (py[i] - entry_py)
                         - position * qty_x * (px[i] - entry_px))
                fees = (leg_fee(entry_py, qty_y) + leg_fee(entry_px, qty_x)
                        + leg_fee(py[i], qty_y) + leg_fee(px[i], qty_x))
                trades.append(gross - fees)
                position = 0
                prev_equity_y = prev_equity_x = 0.0
            continue

        # --- вход ---
        if abs(zi) > config.ENTRY_Z:
            position = -1 if zi > 0 else 1
            entry_py, entry_px = py[i], px[i]
            qty_y = config.CAPITAL_PER_LEG_USDT / entry_py
            qty_x = config.CAPITAL_PER_LEG_USDT / entry_px
            prev_equity_y = prev_equity_x = 0.0

    if not trades:
        return None

    trades = np.array(trades)
    return {
        "y": y_test.name, "x": x_test.name, "pvalue": pval, "beta": beta,
        "half_life_h": hl, "n_trades": len(trades),
        "net_pnl": float(trades.sum()),
        "win_rate": float((trades > 0).mean()),
        "bar_pnl": bar_pnl,
    }


def run(symbols: list[str] | None = None, prices: pd.DataFrame | None = None,
        form_bars: int | None = None, verbose: bool = True) -> dict:
    """Бэктест по всем парам вселенной. Возвращает агрегированные метрики и таблицу пар.

    form_bars — сколько первых баров отвести на формирование (оценку β/mean/σ).
    По умолчанию config.BACKTEST_FORM_BARS.
    """
    if prices is None:
        symbols = symbols or data.top_symbols()
        if verbose:
            print(f"Гружу историю по {len(symbols)} контрактам...")
        prices = data.price_matrix(symbols)
    if verbose:
        print(f"История: {prices.shape[0]} баров × {prices.shape[1]} контрактов.")

    form_n = form_bars if form_bars is not None else config.BACKTEST_FORM_BARS
    form = prices.iloc[:form_n]
    test = prices.iloc[form_n:]
    if len(test) < 24:
        raise ValueError("Слишком мало данных на тестовое окно — увеличь LOOKBACK.")
    if verbose:
        print(f"Formation: {len(form)} баров | Test (out-of-sample): {len(test)} баров")

    from itertools import combinations
    results = []
    equity_bars = pd.Series(0.0, index=test.index)
    for a, b in combinations(prices.columns, 2):
        r = _simulate_pair(form[a], form[b], test[a], test[b])
        if r:
            equity_bars = equity_bars.add(r.pop("bar_pnl"), fill_value=0.0)
            results.append(r)

    df = pd.DataFrame(results)
    equity = equity_bars.cumsum()

    # --- агрегированные метрики ---
    total_pnl = float(equity.iloc[-1]) if len(equity) else 0.0
    peak = equity.cummax()
    max_dd = float((equity - peak).min()) if len(equity) else 0.0
    rets = equity_bars
    sharpe = 0.0
    if rets.std() > 0:
        sharpe = float(rets.mean() / rets.std() * np.sqrt(config.ANNUALIZE_BARS))

    n_trades = int(df["n_trades"].sum()) if not df.empty else 0
    win_rate = float((df["net_pnl"] > 0).mean()) if not df.empty else 0.0

    return {
        "pairs_traded": len(df),
        "n_trades": n_trades,
        "total_pnl_usdt": total_pnl,
        "max_drawdown_usdt": max_dd,
        "sharpe": sharpe,
        "profitable_pairs_share": win_rate,
        "table": df.sort_values("net_pnl", ascending=False) if not df.empty else df,
        "equity": equity,
    }
