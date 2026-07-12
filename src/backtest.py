"""Бэктест стратегии на истории с честным out-of-sample.

Окно делится на две части:
  • FORMATION (первые form_bars свечей): оцениваем β, среднее и σ спреда, проверяем
    коинтеграцию. Торговых решений тут НЕТ.
  • TEST (остаток): торгуем по z-score с ЗАФИКСИРОВАННЫМИ параметрами — измеряем
    стратегию на невиденных данных.

Улучшения против переторговли и издержек (все параметризованы через Params):
  • entry_z / exit_z / stop_z — пороги z-score;
  • cooldown_bars — после выхода не входим N баров (гасит дребезг);
  • max_hold_mult — time-stop: выходим, продержав дольше max_hold_mult × half-life;
  • cost_gate — входим только если ожидаемый возврат (|z|−exit)·σ перекрывает
    комиссию круга (4 ноги) с запасом.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import config
from src import data
from src.cointegration import hedge_ratio, spread, half_life, engle_granger_pvalue


@dataclass
class Params:
    entry_z: float = config.ENTRY_Z
    exit_z: float = config.EXIT_Z
    stop_z: float = config.STOP_Z
    cooldown_bars: int = 0
    max_hold_mult: float = 0.0     # 0 = без time-stop; иначе N×half-life баров
    cost_gate: bool = False
    fee: float = config.TAKER_FEE
    notional: float = config.CAPITAL_PER_LEG_USDT


def _simulate_pair(y_form, x_form, y_test, x_test, p: Params) -> dict | None:
    """Один бэктест пары. Возвращает метрики и ряд PnL по барам теста, либо None."""
    pval = engle_granger_pvalue(y_form, x_form)
    if pval >= config.COINT_PVALUE_MAX:
        return None
    beta = hedge_ratio(y_form, x_form)
    s_form = spread(y_form, x_form, beta)
    mu, sigma = float(s_form.mean()), float(s_form.std())
    hl = half_life(s_form)
    if sigma == 0 or not (config.MIN_HALFLIFE_H <= hl <= config.MAX_HALFLIFE_H):
        return None

    s_test = y_test - beta * x_test
    z = ((s_test - mu) / sigma).values
    py, px = y_test.values, x_test.values
    idx = y_test.index

    max_hold = int(p.max_hold_mult * hl) if p.max_hold_mult else 0
    position = 0
    entry_py = entry_px = qty_y = qty_x = 0.0
    entry_i = -1
    cooldown = 0
    bar_pnl = pd.Series(0.0, index=idx)
    prev_eq = 0.0
    trades = []

    def fee_of(price, qty):
        return price * qty * p.fee

    for i in range(len(z)):
        zi = z[i]
        if position != 0:
            eq = (position * qty_y * (py[i] - entry_py)
                  - position * qty_x * (px[i] - entry_px))
            bar_pnl.iloc[i] = eq - prev_eq
            prev_eq = eq

            held = i - entry_i
            hit_tp = abs(zi) <= p.exit_z
            hit_stop = abs(zi) >= p.stop_z
            hit_time = max_hold and held >= max_hold
            if hit_tp or hit_stop or hit_time:
                fees = (fee_of(entry_py, qty_y) + fee_of(entry_px, qty_x)
                        + fee_of(py[i], qty_y) + fee_of(px[i], qty_x))
                trades.append(eq - fees)
                position = 0
                prev_eq = 0.0
                cooldown = p.cooldown_bars
            continue

        if cooldown > 0:
            cooldown -= 1
            continue

        if abs(zi) > p.entry_z:
            # cost-gate: ожидаемый возврат к среднему должен перекрыть круг комиссий
            if p.cost_gate:
                q_y = p.notional / py[i]
                q_x = p.notional / px[i]
                exp_move = (abs(zi) - p.exit_z) * sigma * q_y   # грубо: движение спреда
                round_fee = 2 * (fee_of(py[i], q_y) + fee_of(px[i], q_x))
                if exp_move <= round_fee * 1.5:
                    continue
            position = -1 if zi > 0 else 1
            entry_py, entry_px, entry_i = py[i], px[i], i
            qty_y = p.notional / entry_py
            qty_x = p.notional / entry_px
            prev_eq = 0.0

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
        form_bars: int | None = None, params: Params | None = None,
        verbose: bool = True) -> dict:
    """Бэктест по всем парам вселенной. Возвращает агрегированные метрики и таблицу пар."""
    p = params or Params()
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

    results = []
    equity_bars = pd.Series(0.0, index=test.index)
    for a, b in combinations(prices.columns, 2):
        r = _simulate_pair(form[a], form[b], test[a], test[b], p)
        if r:
            equity_bars = equity_bars.add(r.pop("bar_pnl"), fill_value=0.0)
            results.append(r)

    df = pd.DataFrame(results)
    equity = equity_bars.cumsum()
    total_pnl = float(equity.iloc[-1]) if len(equity) else 0.0
    peak = equity.cummax()
    max_dd = float((equity - peak).min()) if len(equity) else 0.0
    sharpe = 0.0
    if equity_bars.std() > 0:
        sharpe = float(equity_bars.mean() / equity_bars.std() * np.sqrt(config.ANNUALIZE_BARS))
    n_trades = int(df["n_trades"].sum()) if not df.empty else 0
    win_rate = float((df["net_pnl"] > 0).mean()) if not df.empty else 0.0

    return {
        "pairs_traded": len(df), "n_trades": n_trades,
        "total_pnl_usdt": total_pnl, "max_drawdown_usdt": max_dd,
        "sharpe": sharpe, "profitable_pairs_share": win_rate,
        "table": df.sort_values("net_pnl", ascending=False) if not df.empty else df,
        "equity": equity,
    }
