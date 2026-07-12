"""Исполнение ордеров: paper (симуляция) и live (реальные ордера BingX).

Единый интерфейс Executor. Бот работает с ним, не зная, симуляция это или биржа.
Смена MODE в config.py переключает реализацию — код бота не меняется.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass

import requests

import config
from src import data


@dataclass
class Fill:
    symbol: str
    side: str          # "BUY" или "SELL"
    qty: float
    price: float
    fee: float         # комиссия в USDT


class Executor:
    """Базовый интерфейс исполнителя."""

    def market_order(self, symbol: str, side: str, qty: float) -> Fill:
        raise NotImplementedError

    def price(self, symbol: str) -> float:
        return data.latest_price(symbol)


class PaperExecutor(Executor):
    """Симуляция: исполнение по текущей рыночной цене с учётом комиссии тейкера.

    Реальных ордеров не отправляет — деньги в безопасности. Идеально для проверки
    стратегии на живых данных (paper trading), как советует и сама статья.
    """

    def market_order(self, symbol: str, side: str, qty: float) -> Fill:
        px = self.price(symbol)
        fee = px * qty * config.TAKER_FEE
        return Fill(symbol=symbol, side=side, qty=qty, price=px, fee=fee)


class BingXExecutor(Executor):
    """Реальные ордера на BingX perpetual. Требует BINGX_API_KEY / BINGX_API_SECRET.

    ВНИМАНИЕ: отправляет настоящие сделки с реальными деньгами. Включается только
    при MODE='live' и заданных ключах. Перед использованием проверьте на тестовых
    объёмах и убедитесь, что понимаете размер плеча и риски.
    """

    def __init__(self) -> None:
        if not config.BINGX_API_KEY or not config.BINGX_API_SECRET:
            raise RuntimeError(
                "MODE='live', но BINGX_API_KEY/BINGX_API_SECRET не заданы в окружении."
            )

    def _sign(self, params: dict) -> str:
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        sig = hmac.new(
            config.BINGX_API_SECRET.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return f"{query}&signature={sig}"

    def market_order(self, symbol: str, side: str, qty: float) -> Fill:
        ts = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "side": side,               # BUY / SELL
            "positionSide": "LONG" if side == "BUY" else "SHORT",
            "type": "MARKET",
            "quantity": qty,
            "timestamp": ts,
        }
        url = f"{config.BINGX_BASE}/openApi/swap/v2/trade/order?{self._sign(params)}"
        r = requests.post(url, headers={"X-BX-APIKEY": config.BINGX_API_KEY}, timeout=20)
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"BingX order error: {resp}")
        # Цену исполнения берём текущую (для точности можно опросить ордер по orderId)
        px = self.price(symbol)
        fee = px * qty * config.TAKER_FEE
        return Fill(symbol=symbol, side=side, qty=qty, price=px, fee=fee)


def make_executor() -> Executor:
    """Фабрика по config.MODE."""
    if config.MODE == "live":
        return BingXExecutor()
    return PaperExecutor()
