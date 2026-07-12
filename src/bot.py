"""Главный цикл бота статистического арбитража.

Каждый шаг:
  1) (периодически) заново находит коинтегрированные пары — watchlist;
  2) тянет текущие цены нужных контрактов;
  3) закрывает открытые позиции по take-profit (|z|<=EXIT) или стопу (|z|>=STOP);
  4) открывает новые по сигналу |z|>ENTRY, соблюдая лимит MAX_OPEN_POSITIONS.

Направление сделки:
  z > +ENTRY  → спред слишком высок → SHORT spread (sell y, buy x), direction=-1
  z < -ENTRY  → спред слишком низок → LONG  spread (buy y, sell x), direction=+1
Выход всегда при возврате к среднему (|z|<=EXIT) или разрыве коинтеграции (|z|>=STOP).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import config
from src import data
from src.executor import make_executor
from src.portfolio import Portfolio
from src.screener import screen
from src.cointegration import live_zscore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line)
    try:
        with open(config.LOG_FILE, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Bot:
    def __init__(self) -> None:
        self.ex = make_executor()
        self.pf = Portfolio()
        self.watchlist: list[dict] = []
        self._last_detect_ts = 0.0

    # --- Обнаружение пар ---
    def detect(self) -> None:
        _log("Поиск коинтегрированных пар...")
        symbols = data.top_symbols()
        result = screen(symbols)
        if result.empty:
            _log("  пар не найдено при текущих порогах.")
            self.watchlist = []
        else:
            # верхние кандидаты; ноги открытых позиций всегда оставляем в списке
            self.watchlist = result.to_dict("records")
            _log(f"  найдено пар: {len(self.watchlist)}. Топ: "
                 + ", ".join(f"{r['y']}/{r['x']}(p={r['pvalue']:.3f})"
                             for r in self.watchlist[:5]))
        self._last_detect_ts = time.time()

    # --- Один торговый шаг ---
    def step(self) -> None:
        # символы, по которым нужны цены: watchlist + открытые позиции
        needed = set()
        for r in self.watchlist:
            needed.update((r["y"], r["x"]))
        for pos in self.pf.positions.values():
            needed.update((pos.y, pos.x))
        if not needed:
            _log("Нет пар для мониторинга.")
            return
        prices = data.latest_prices(sorted(needed))

        # 1) управление открытыми позициями
        for pos in list(self.pf.positions.values()):
            if pos.y not in prices or pos.x not in prices:
                continue
            z = pos.current_z(prices[pos.y], prices[pos.x])
            upnl = pos.unrealized_pnl(prices[pos.y], prices[pos.x])
            if abs(z) <= config.EXIT_Z:
                net = self.pf.close_pair(self.ex, pos, _now())
                _log(f"CLOSE {pos.y}/{pos.x} take-profit z={z:.2f} PnL={net:+.2f} USDT")
            elif abs(z) >= config.STOP_Z:
                net = self.pf.close_pair(self.ex, pos, _now())
                _log(f"CLOSE {pos.y}/{pos.x} STOP z={z:.2f} PnL={net:+.2f} USDT")
            else:
                _log(f"HOLD  {pos.y}/{pos.x} z={z:.2f} uPnL={upnl:+.2f}")

        # 2) открытие новых по сигналу
        for r in self.watchlist:
            if len(self.pf.positions) >= config.MAX_OPEN_POSITIONS:
                break
            if self.pf.is_open(r["y"], r["x"]):
                continue
            if r["y"] not in prices or r["x"] not in prices:
                continue
            z = live_zscore(
                prices[r["y"]], prices[r["x"]], r["beta"], r["spread_mean"], r["spread_std"]
            )
            if abs(z) > config.ENTRY_Z:
                direction = -1 if z > 0 else 1   # высокий спред → short; низкий → long
                pos = self.pf.open_pair(
                    self.ex, r["y"], r["x"], r["beta"], r["spread_mean"],
                    r["spread_std"], direction, z, _now(),
                )
                side = "SHORT spread" if direction < 0 else "LONG spread"
                _log(f"OPEN  {r['y']}/{r['x']} {side} z={z:.2f} "
                     f"qty=({pos.qty_y:.4f},{pos.qty_x:.4f})")

        # 3) сводка
        upnl = self.pf.total_unrealized(prices)
        _log(f"Итог: открыто {len(self.pf.positions)} | реализовано "
             f"{self.pf.realized_pnl:+.2f} | плавающий {upnl:+.2f} | "
             f"сделок закрыто {self.pf.closed_trades}")

    # --- Бесконечный цикл ---
    def run(self) -> None:
        _log(f"=== Старт бота | режим={config.MODE.upper()} | "
             f"капитал/нога={config.CAPITAL_PER_LEG_USDT} USDT | "
             f"макс.позиций={config.MAX_OPEN_POSITIONS} ===")
        if config.MODE != "live":
            _log("PAPER-режим: сделки симулируются, реальные деньги не задействованы.")
        self.detect()
        while True:
            try:
                if (time.time() - self._last_detect_ts) >= config.REDETECT_EVERY_MIN * 60:
                    self.detect()
                self.step()
            except Exception as e:  # noqa: BLE001 — цикл не должен падать из-за разовой ошибки
                _log(f"ОШИБКА шага: {e}")
            time.sleep(config.POLL_INTERVAL_SEC)
