#!/usr/bin/env python3
"""Запуск бота статистического арбитража на BingX.

    python run_bot.py            # бесконечный цикл (режим из config.MODE)
    python run_bot.py --once     # один шаг (детект + одна проверка сигналов) и выход

Режим paper/live и все пороги — в config.py. Для live нужны переменные окружения
BINGX_API_KEY / BINGX_API_SECRET.
"""
from __future__ import annotations

import sys

from src.bot import Bot


def main() -> None:
    once = "--once" in sys.argv
    bot = Bot()
    if once:
        bot.detect()
        bot.step()
    else:
        bot.run()


if __name__ == "__main__":
    main()
