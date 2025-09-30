import logging
import pandas as pd
import ccxt
import os
import numpy as np
import pytz
import ta
from telegram import constants
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler

# ================== CONFIG ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
try:
    CHAT_ID = int(CHAT_ID) if CHAT_ID else None
except ValueError:
    print(f"⚠️ Invalid CHAT_ID value: {CHAT_ID}")
    CHAT_ID = None

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "BNB/USDT", "SOL/USDT",
    "DOGE/USDT", "TRX/USDT", "ADA/USDT", "HYPE/USDT",
    "LINK/USDT", "AVAX/USDT", "XLM/USDT", "SUI/USDT"
]
TIMEFRAMES = ["1h", "4h", "1d"]
LIMIT = 200
CHECK_INTERVAL_HOURS = 1
# ============================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

exchange = ccxt.binance()
last_signals = {s: {tf: None for tf in TIMEFRAMES} for s in SYMBOLS}

# ---------------- INDICATOR FUNCTIONS ----------------
def fetch_ohlcv(symbol, timeframe, limit=LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def ichimoku(df):
    high_9 = df["high"].rolling(9).max()
    low_9 = df["low"].rolling(9).min()
    df["tenkan_sen"] = (high_9 + low_9) / 2

    high_26 = df["high"].rolling(26).max()
    low_26 = df["low"].rolling(26).min()
    df["kijun_sen"] = (high_26 + low_26) / 2

    high_52 = df["high"].rolling(52).max()
    low_52 = df["low"].rolling(52).min()
    df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)
    df["senkou_span_b"] = ((high_52 + low_52) / 2).shift(26)

    df["chikou_span"] = df["close"].shift(-26)

    # ✅ RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    return df

def analyze_ichimoku(df):
    latest = df.iloc[-1]

    cloud_top = max(latest["senkou_span_a"], latest["senkou_span_b"])
    cloud_bottom = min(latest["senkou_span_a"], latest["senkou_span_b"])

    lagging = latest["chikou_span"]
    close = latest["close"]
    rsi = latest["rsi"]
    signal = "Neutral"
    lagging_info = ""
    sl, tp = None, None

    # ✅ Buy: Close above cloud + Tenkan > Kijun + Lagging above cloud
    if close > cloud_top and latest["tenkan_sen"] > latest["kijun_sen"] and lagging > cloud_top:
        signal = "BUY"
        sl = cloud_bottom
