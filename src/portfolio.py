"""Учёт позиций и PnL. Позиция = пара из двух ног (long одной, short другой)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field

import config
from src.executor import Executor, Fill
from src.cointegration import live_zscore


@dataclass
class PairPosition:
    y: str
    x: str
    beta: float
    spread_mean: float
    spread_std: float
    direction: int          # +1 = long spread (buy y, sell x); -1 = short spread
    qty_y: float
    qty_x: float
    entry_price_y: float
    entry_price_x: float
    entry_z: float
    entry_fee: float
    opened_at: str

    def unrealized_pnl(self, price_y: float, price_x: float) -> float:
        """Плавающий PnL по паре при текущих ценах ног (без учёта комиссии закрытия)."""
        # long spread: +y, -x ; short spread: -y, +x
        pnl_y = self.direction * self.qty_y * (price_y - self.entry_price_y)
        pnl_x = -self.direction * self.qty_x * (price_x - self.entry_price_x)
        return pnl_y + pnl_x

    def current_z(self, price_y: float, price_x: float) -> float:
        return live_zscore(price_y, price_x, self.beta, self.spread_mean, self.spread_std)


class Portfolio:
    """Открытые позиции, реализованный PnL, персист состояния на диск."""

    def __init__(self) -> None:
        self.positions: dict[str, PairPosition] = {}
        self.realized_pnl: float = 0.0
        self.closed_trades: int = 0
        self.load()

    @staticmethod
    def key(y: str, x: str) -> str:
        return f"{y}|{x}"

    # --- Открытие / закрытие ---
    def open_pair(self, ex: Executor, y: str, x: str, beta: float,
                  spread_mean: float, spread_std: float, direction: int,
                  z: float, now: str) -> PairPosition:
        """Открывает пару: две рыночные ноги. direction=+1 long spread, -1 short spread."""
        # Номинал на ногу → количество контрактов = notional / цена
        px_y = ex.price(y)
        px_x = ex.price(x)
        qty_y = config.CAPITAL_PER_LEG_USDT / px_y
        qty_x = config.CAPITAL_PER_LEG_USDT / px_x

        # long spread: BUY y, SELL x ; short spread: SELL y, BUY x
        side_y = "BUY" if direction > 0 else "SELL"
        side_x = "SELL" if direction > 0 else "BUY"
        fill_y: Fill = ex.market_order(y, side_y, qty_y)
        fill_x: Fill = ex.market_order(x, side_x, qty_x)

        pos = PairPosition(
            y=y, x=x, beta=beta, spread_mean=spread_mean, spread_std=spread_std,
            direction=direction, qty_y=qty_y, qty_x=qty_x,
            entry_price_y=fill_y.price, entry_price_x=fill_x.price,
            entry_z=z, entry_fee=fill_y.fee + fill_x.fee, opened_at=now,
        )
        self.positions[self.key(y, x)] = pos
        self.save()
        return pos

    def close_pair(self, ex: Executor, pos: PairPosition, now: str) -> float:
        """Закрывает пару (реверс обеих ног). Возвращает реализованный PnL за вычетом комиссий."""
        side_y = "SELL" if pos.direction > 0 else "BUY"
        side_x = "BUY" if pos.direction > 0 else "SELL"
        fill_y = ex.market_order(pos.y, side_y, pos.qty_y)
        fill_x = ex.market_order(pos.x, side_x, pos.qty_x)

        gross = pos.unrealized_pnl(fill_y.price, fill_x.price)
        fees = pos.entry_fee + fill_y.fee + fill_x.fee
        net = gross - fees

        self.realized_pnl += net
        self.closed_trades += 1
        del self.positions[self.key(pos.y, pos.x)]
        self.save()
        return net

    # --- Метрики ---
    def total_unrealized(self, prices: dict[str, float]) -> float:
        total = 0.0
        for pos in self.positions.values():
            if pos.y in prices and pos.x in prices:
                total += pos.unrealized_pnl(prices[pos.y], prices[pos.x])
        return total

    def is_open(self, y: str, x: str) -> bool:
        return self.key(y, x) in self.positions

    # --- Персист ---
    def save(self) -> None:
        state = {
            "realized_pnl": self.realized_pnl,
            "closed_trades": self.closed_trades,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
        }
        with open(config.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def load(self) -> None:
        if not os.path.exists(config.STATE_FILE):
            return
        with open(config.STATE_FILE) as f:
            state = json.load(f)
        self.realized_pnl = state.get("realized_pnl", 0.0)
        self.closed_trades = state.get("closed_trades", 0)
        self.positions = {
            k: PairPosition(**v) for k, v in state.get("positions", {}).items()
        }
