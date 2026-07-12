"""Параметры бота когерентного (статистического) арбитража. Биржа: BingX (perpetual futures)."""

# --- Источник данных (BingX public API, ключи не нужны для котировок) ---
BINGX_BASE = "https://open-api.bingx.com"
QUOTE = "USDT"                # котируем всё в USDT; символы вида BTC-USDT
INTERVAL = "1h"               # таймфрейм свечей
LOOKBACK = 720                # свечей истории (720ч ≈ 30 дней) — окно оценки коинтеграции

# --- Формирование вселенной пар ---
TOP_N_BY_VOLUME = 30          # брать N самых ликвидных контрактов по обороту
EXCLUDE = {                   # стейблы и обёртки — бессмысленны для арбитража
    "USDC-USDT", "FDUSD-USDT", "TUSD-USDT", "DAI-USDT", "USDP-USDT",
}

# --- Пороги коинтеграции / отбора пар ---
COINT_PVALUE_MAX = 0.05       # тест Энгла-Грейнджера: p-value < 0.05 → коинтеграция
MIN_HALFLIFE_H = 1            # период полураспада спреда, часы (быстрее — лучше)
MAX_HALFLIFE_H = 240          # медленнее 10 дней — возврат слишком долгий, отбрасываем
ZSCORE_WINDOW = 0             # 0 = z-score по всей выборке; >0 = скользящее окно (свечей)

# --- Торговые пороги (z-score спреда) ---
ENTRY_Z = 2.0                 # вход при |Z| > 2
EXIT_Z = 0.5                  # выход при |Z| <= 0.5 (take profit — возврат к среднему)
STOP_Z = 3.5                  # стоп при |Z| > 3.5 (коинтеграция ломается)

# --- Бэктест ---
BACKTEST_FORM_BARS = 480      # свечей на формирование (оценка β/mean/σ), out-of-sample — остаток
ANNUALIZE_BARS = 24 * 365     # часовых баров в году (для Sharpe)

# --- Параметры бота ---
MODE = "paper"                # "paper" (симуляция) или "live" (реальные ордера, нужны ключи)
POLL_INTERVAL_SEC = 60        # как часто опрашивать цены и проверять сигналы
REDETECT_EVERY_MIN = 360      # как часто заново искать коинтегрированные пары (6ч)
CAPITAL_PER_LEG_USDT = 100.0  # номинал на одну ногу сделки (обе ноги ≈ 2×)
MAX_OPEN_POSITIONS = 5        # максимум одновременно открытых пар
TAKER_FEE = 0.0005            # комиссия тейкера BingX (0.05%) — учитывается в paper-PnL
STATE_FILE = "bot_state.json" # файл состояния (открытые позиции, PnL)
LOG_FILE = "trades.log"       # журнал сделок

# --- Ключи API (только для MODE="live"). НЕ храните в git! Берутся из окружения: ---
#   export BINGX_API_KEY=...    export BINGX_API_SECRET=...
import os
BINGX_API_KEY = os.getenv("BINGX_API_KEY", "")
BINGX_API_SECRET = os.getenv("BINGX_API_SECRET", "")
