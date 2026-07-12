"""Загрузка рыночных данных с публичного API BingX (perpetual futures, без ключей)."""
from __future__ import annotations

import time
import requests
import pandas as pd

import config


def _get(path: str, params: dict | None = None) -> object:
    url = f"{config.BINGX_BASE}{path}"
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("code") not in (0, None):
                raise RuntimeError(f"BingX error {payload.get('code')}: {payload.get('msg')}")
            return payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        except (requests.RequestException, RuntimeError):
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def top_symbols(n: int | None = None) -> list[str]:
    """Топ-N ликвидных перпетуальных контрактов к QUOTE по обороту за 24ч."""
    n = n or config.TOP_N_BY_VOLUME
    data = _get("/openApi/swap/v2/quote/ticker")
    rows = [
        d for d in data
        if d["symbol"].endswith(f"-{config.QUOTE}")
        and d["symbol"] not in config.EXCLUDE
    ]
    rows.sort(key=lambda d: float(d.get("quoteVolume") or 0), reverse=True)
    return [d["symbol"] for d in rows[:n]]


def klines(symbol: str, interval: str | None = None, limit: int | None = None) -> pd.Series:
    """Ряд цен закрытия свечей (индекс — время открытия свечи, UTC, по возрастанию)."""
    interval = interval or config.INTERVAL
    limit = limit or config.LOOKBACK
    raw = _get(
        "/openApi/swap/v3/quote/klines",
        {"symbol": symbol, "interval": interval, "limit": limit},
    )
    # BingX отдаёт свечи по убыванию времени — сортируем по возрастанию
    raw = sorted(raw, key=lambda c: int(c["time"]))
    idx = pd.to_datetime([int(c["time"]) for c in raw], unit="ms", utc=True)
    close = pd.Series(
        [float(c["close"]) for c in raw], index=idx, name=symbol, dtype="float64"
    )
    return close


def klines_n(symbol: str, interval: str, n_bars: int, page: int = 1440) -> pd.Series:
    """Как klines, но с пагинацией: собирает n_bars свечей несколькими запросами.

    BingX отдаёт максимум ~1440 свечей за раз. Идём назад по времени через endTime,
    пока не наберём нужное количество. Возвращает ряд цен закрытия (UTC, по возрастанию).
    """
    collected: dict[int, float] = {}
    end_time: int | None = None
    guard = 0
    while len(collected) < n_bars and guard < 200:
        guard += 1
        params = {"symbol": symbol, "interval": interval, "limit": page}
        if end_time is not None:
            params["endTime"] = end_time
        raw = _get("/openApi/swap/v3/quote/klines", params)
        if not raw:
            break
        for c in raw:
            collected[int(c["time"])] = float(c["close"])
        oldest = min(int(c["time"]) for c in raw)
        if end_time is not None and oldest >= end_time:
            break  # не двигаемся дальше — данных больше нет
        end_time = oldest - 1
        time.sleep(0.25)  # бережём rate-limit
    items = sorted(collected.items())[-n_bars:]
    idx = pd.to_datetime([t for t, _ in items], unit="ms", utc=True)
    return pd.Series([v for _, v in items], index=idx, name=symbol, dtype="float64")


def price_matrix_n(symbols: list[str], interval: str, n_bars: int,
                   min_coverage: float = 0.9) -> pd.DataFrame:
    """price_matrix с заданным интервалом и числом баров (через пагинацию)."""
    series = {}
    for sym in symbols:
        try:
            series[sym] = klines_n(sym, interval, n_bars)
        except Exception as e:  # noqa: BLE001
            print(f"  ! пропуск {sym}: {e}")
    df = pd.DataFrame(series)
    min_len = int(min_coverage * n_bars)
    enough = [c for c in df.columns if df[c].notna().sum() >= min_len]
    df = df[enough].dropna(how="any")
    return df


def latest_price(symbol: str) -> float:
    """Текущая цена контракта."""
    data = _get("/openApi/swap/v2/quote/price", {"symbol": symbol})
    return float(data["price"])


def latest_prices(symbols: list[str]) -> dict[str, float]:
    """Текущие цены для набора символов (по одному запросу — BingX не даёт батч price)."""
    out = {}
    for s in symbols:
        try:
            out[s] = latest_price(s)
        except Exception as e:  # noqa: BLE001
            print(f"  ! цена {s}: {e}")
    return out


def price_matrix(symbols: list[str], min_coverage: float = 0.9) -> pd.DataFrame:
    """Матрица цен закрытия: строки — время, столбцы — символы, выровнена по общему времени.

    Монеты с короткой историей (недавние листинги) отбрасываются, иначе общий
    dropna схлопнул бы окно. Оставляем символы с покрытием >= min_coverage от LOOKBACK.
    """
    series = {}
    for sym in symbols:
        try:
            series[sym] = klines(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  ! пропуск {sym}: {e}")
    df = pd.DataFrame(series)

    min_len = int(min_coverage * config.LOOKBACK)
    enough = [c for c in df.columns if df[c].notna().sum() >= min_len]
    dropped = [c for c in df.columns if c not in enough]
    if dropped:
        print(f"  отброшены по короткой истории: {', '.join(dropped)}")
    df = df[enough].dropna(how="any")
    return df
