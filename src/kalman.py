"""Динамический хедж через фильтр Калмана (pairs trading, подход Chan).

Наивная стратегия проваливалась OOS потому, что β (коэффициент хеджа) оценивался
ОДИН раз на formation и «уплывал». Здесь β оценивается ОНЛАЙН на каждом баре как
скрытое состояние случайного блуждания:

    state:  β_t = [intercept_t, slope_t] = β_{t-1} + w_t,   w ~ N(0, Vw)
    obs:    P_y,t = [1, P_x,t] · β_t + v_t,                 v ~ N(0, Ve)

Ошибка прогноза наблюдения e_t = P_y,t − [1,P_x,t]·β_{t-1} — это «мгновенный спред»,
а её дисперсия Q_t известна из фильтра. Сигнал = e_t/√Q_t (стандартизованная
инновация) — самонастраивающийся аналог z-score. Всё КАУЗАЛЬНО: β_{t-1} и Q_t на
шаге t используют только прошлое, поэтому весь ряд после burn-in — честный OOS.

Vw = δ/(1−δ)·I управляет скоростью адаптации β (больше δ → быстрее плывёт),
Ve — дисперсия наблюдения. Значения по умолчанию — из Chan (не подгонка под данные).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge(y: np.ndarray, x: np.ndarray,
                 delta: float = 1e-4, r_var: float = 1e-3) -> dict:
    """Онлайн-оценка β_t и стандартизованной инновации по паре (y на x).

    Возвращает массивы одинаковой длины:
        intercept, slope — состояние β_t на каждом баре;
        e — ошибка прогноза (мгновенный спред), каузальная;
        sqrt_Q — стандартное отклонение инновации;
        z = e/√Q — торговый сигнал.
    Реализация развёрнута в скаляры (2×2 вручную) — на порядок быстрее numpy-матриц
    в горячем цикле по десяткам тысяч баров.
    """
    n = len(y)
    q = delta / (1.0 - delta)          # дисперсия блуждания состояния (Vw = q·I)
    ve = r_var

    a = 0.0                            # intercept состояния
    b = 0.0                            # slope состояния (коэффициент хеджа)
    p00 = p01 = p11 = 0.0              # ковариация состояния P (симметрична)

    slope = np.empty(n)
    intercept = np.empty(n)
    e = np.empty(n)
    sqrt_q = np.empty(n)

    for t in range(n):
        xt = x[t]
        # прогноз ковариации: R = P + Vw
        r00 = p00 + q
        r01 = p01
        r11 = p11 + q
        # ошибка прогноза наблюдения (по β_{t-1})
        yhat = a + b * xt
        et = y[t] - yhat
        # R·Xt  и  Qt = Xt·R·Xt + Ve
        rx0 = r00 + r01 * xt
        rx1 = r01 + r11 * xt
        qt = rx0 + rx1 * xt + ve
        # калмановский коэффициент K = R·Xt / Qt
        k0 = rx0 / qt
        k1 = rx1 / qt
        # обновление состояния
        a += k0 * et
        b += k1 * et
        # обновление ковариации: P = R − K·(Xt·R)
        p00 = r00 - k0 * rx0
        p01 = r01 - k0 * rx1
        p11 = r11 - k1 * rx1

        intercept[t] = a
        slope[t] = b
        e[t] = et
        sqrt_q[t] = np.sqrt(qt)

    z = e / sqrt_q
    return {"intercept": intercept, "slope": slope, "e": e, "sqrt_Q": sqrt_q, "z": z}


def simulate_kalman(py: np.ndarray, px: np.ndarray, *,
                    delta: float = 1e-4, r_var: float = 1e-3,
                    entry_z: float = 1.5, exit_z: float = 0.5, stop_z: float = 4.0,
                    burn_in: int = 1000, fee: float = 0.0005,
                    notional: float = 100.0, signal_window: int = 120) -> dict | None:
    """Каузальный бэктест пары на калмановском сигнале (в LOG-ценах).

    Фильтр работает по log-ценам: тогда slope β — безразмерная эластичность
    (dlog y = β·dlog x), а хедж экономически осмыслен. PnL считается по РЕАЛЬНЫМ
    доходностям с долларово-нейтральным хеджем: экспозиция $notional на ногу y и
    $β·notional на ногу x (обе ноги сопоставимы, а не «$100 против 7 центов»).

    Сделки открываются только ПОСЛЕ burn_in. Весь торгуемый участок — честный OOS.
    Возвращает метрики и по-баровый PnL, либо None.
    """
    n = len(py)
    if n <= burn_in + 10:
        return None
    ly = np.log(py)
    lx = np.log(px)
    kf = kalman_hedge(ly, lx, delta=delta, r_var=r_var)
    slope = kf["slope"]           # эластичность β в log-пространстве
    # Сигнал: инновация e, нормированная СВОИМ скользящим std (каузально, по прошлому).
    # Это устраняет зависимость от калибровки r_var (Q), которая в log-пространстве
    # некорректна. std берём со сдвигом на 1 бар — будущее не подглядываем.
    e = pd.Series(kf["e"])
    roll_std = e.rolling(signal_window, min_periods=signal_window // 2).std().shift(1)
    z = (kf["e"] / roll_std.values)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    position = 0
    dollar_y = dollar_x = 0.0     # долларовые экспозиции ног, фиксируются на входе
    trade_gross = 0.0             # рыночный PnL сделки БЕЗ комиссий
    trade_cost = 0.0             # комиссии сделки (вход+выход)
    bar_pnl = np.zeros(n)
    trades: list[float] = []      # net (gross − комиссии)
    gross_trades: list[float] = []
    total_fees = 0.0

    for i in range(burn_in, n):
        zi = z[i]
        if position != 0:
            ry = py[i] / py[i - 1] - 1.0
            rx = px[i] / px[i - 1] - 1.0
            pnl_i = position * (dollar_y * ry - dollar_x * rx)
            bar_pnl[i] = pnl_i
            trade_gross += pnl_i
            if abs(zi) <= exit_z or abs(zi) >= stop_z:
                exit_fee = fee * (dollar_y + abs(dollar_x))   # выход по обеим ногам
                trade_cost += exit_fee
                trades.append(trade_gross - trade_cost)
                gross_trades.append(trade_gross)
                total_fees += trade_cost
                position = 0
            continue

        if entry_z < abs(zi) < stop_z:
            position = -1 if zi > 0 else 1     # zi>0: y дорог → шорт спреда
            dollar_y = notional
            dollar_x = slope[i] * notional
            entry_fee = fee * (dollar_y + abs(dollar_x))
            trade_gross = 0.0
            trade_cost = entry_fee             # комиссия входа
            bar_pnl[i] += -entry_fee

    if not trades:
        return None
    tr = np.array(trades)
    return {
        "n_trades": len(tr),
        "net_pnl": float(tr.sum()),
        "gross_pnl": float(np.sum(gross_trades)),
        "total_fees": float(total_fees),
        "win_rate": float((tr > 0).mean()),
        "bar_pnl": bar_pnl,
    }
