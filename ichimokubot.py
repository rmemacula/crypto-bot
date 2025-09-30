import logging
import pandas as pd
import ccxt
import os
import numpy as np
import pytz
import ta
import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

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
        tp = close + 2 * (close - sl)

    # ✅ Sell: Close below cloud + Tenkan < Kijun + Lagging below cloud
    elif close < cloud_bottom and latest["tenkan_sen"] < latest["kijun_sen"] and lagging < cloud_bottom:
        signal = "SELL"
        sl = cloud_top
        tp = close - 2 * (sl - close)

    # Lagging Span info
    if not np.isnan(lagging):
        if lagging > cloud_top:
            lagging_info = "(Lagging ABOVE cloud)"
        elif lagging < cloud_bottom:
            lagging_info = "(Lagging BELOW cloud)"
        else:
            lagging_info = "(Lagging INSIDE cloud)"

    return signal, close, lagging_info, rsi, sl, tp

# ---------------- ALERT FUNCTION -------------------
async def check_and_alert(context: ContextTypes.DEFAULT_TYPE):
    global last_signals
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, tf)
                df = ichimoku(df)
                signal, price, lagging_info, rsi, sl, tp = analyze_ichimoku(df)
                prev_signal = last_signals[symbol][tf]

                if signal != "Neutral" and signal != prev_signal:
                    msg = (
                        f"🚨 {symbol} ({tf})\n"
                        f"Signal: {signal} {lagging_info}\n"
                        f"RSI: {rsi:.1f}\n"
                        f"Entry: {price:.4f} USDT"
                    )
                    if sl and tp:
                        msg += f"\nSL: {sl:.4f} USDT\nTP: {tp:.4f} USDT"

                    await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                    last_signals[symbol][tf] = signal

            except Exception as e:
                logger.error(f"Error fetching {symbol} {tf}: {e}")

# ---------------- TELEGRAM COMMANDS -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Multi-coin Ichimoku bot online ✅")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg_list = []
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, tf)
                df = ichimoku(df)
                signal, price, lagging_info, rsi, sl, tp = analyze_ichimoku(df)
                line = f"{symbol} ({tf}): {signal} {lagging_info} | RSI {rsi:.1f} @ {price:.4f} USDT"
                if signal in ["BUY", "SELL"]:
                    line += f" | SL {sl:.2f} | TP {tp:.2f}"
                msg_list.append(line)
            except:
                msg_list.append(f"{symbol} ({tf}): Error fetching")
    await update.message.reply_text("\n".join(msg_list))

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is online and working!")

# ---------------- HEARTBEAT -------------------
async def heartbeat(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=CHAT_ID, text="🤖 Ichimoku bot heartbeat: still running, waiting for signals...")

# ---------------- MAIN -------------------
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("test", test))

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=pytz.UTC)
    scheduler.add_job(check_and_alert, "interval", hours=CHECK_INTERVAL_HOURS, args=[application.bot])
    scheduler.add_job(heartbeat, "interval", hours=6, args=[application.bot])
    scheduler.start()

    application.run_polling()

if __name__ == "__main__":
    main()
