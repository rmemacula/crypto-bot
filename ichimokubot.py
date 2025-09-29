import ccxt
import pandas as pd
import numpy as np
import ta
import os
import pytz
from datetime import datetime
from telegram import Bot
from apscheduler.schedulers.background import BackgroundScheduler

# =========================
# CONFIG
# =========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

exchange = ccxt.binance()

symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]  # Add coins here
timeframes = ["1h", "4h", "1d"]  # All timeframes to check

# =========================
# FUNCTIONS
# =========================
def fetch_ohlcv(symbol, timeframe="1h", limit=150):
    try:
        data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
        return None

def analyze_ichimoku(df):
    # Ichimoku components
    high9 = df["high"].rolling(window=9).max()
    low9 = df["low"].rolling(window=9).min()
    conversion_line = (high9 + low9) / 2

    high26 = df["high"].rolling(window=26).max()
    low26 = df["low"].rolling(window=26).min()
    base_line = (high26 + low26) / 2

    span_a = ((conversion_line + base_line) / 2).shift(26)
    high52 = df["high"].rolling(window=52).max()
    low52 = df["low"].rolling(window=52).min()
    span_b = ((high52 + low52) / 2).shift(26)

    lagging_span = df["close"].shift(-26)

    latest = df.iloc[-1]
    price = latest["close"]

    signal = "NEUTRAL"
    stop_loss = None
    take_profit = None

    # Buy condition
    if conversion_line.iloc[-1] > base_line.iloc[-1] and price > span_a.iloc[-1] and price > span_b.iloc[-1]:
        signal = "BUY"
        # SL below cloud
        stop_loss = min(span_a.iloc[-1], span_b.iloc[-1])
        risk = price - stop_loss
        take_profit = price + 2 * risk

    # Sell condition
    elif conversion_line.iloc[-1] < base_line.iloc[-1] and price < span_a.iloc[-1] and price < span_b.iloc[-1]:
        signal = "SELL"
        # SL above cloud
        stop_loss = max(span_a.iloc[-1], span_b.iloc[-1])
        risk = stop_loss - price
        take_profit = price - 2 * risk

    return signal, price, stop_loss, take_profit, conversion_line.iloc[-1], base_line.iloc[-1], lagging_span.iloc[-1]

def check_signals():
    for symbol in symbols:
        for tf in timeframes:
            df = fetch_ohlcv(symbol, tf)
            if df is None:
                continue
            signal, price, sl, tp, tenkan, kijun, chikou = analyze_ichimoku(df)

            if signal in ["BUY", "SELL"]:
                message = (
                    f"🚨 *{signal} Signal Detected*\n"
                    f"📊 {symbol} | TF: {tf}\n"
                    f"💵 Price: {price:.2f}\n"
                    f"🔵 Tenkan: {tenkan:.2f} | 🔴 Kijun: {kijun:.2f}\n"
                    f"⏳ Lagging: {chikou:.2f}\n"
                    f"❌ Stop Loss: {sl:.2f}\n"
                    f"✅ Take Profit: {tp:.2f}\n"
                    f"📈 R:R = 1:2"
                )
                bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

# =========================
# SCHEDULER
# =========================
scheduler = BackgroundScheduler(timezone=pytz.utc)
scheduler.add_job(check_signals, "interval", minutes=60)  # Every 1h
scheduler.start()

print("Bot started... waiting for signals 🚀")
