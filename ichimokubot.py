import logging
import pandas as pd
import ccxt
import os
import numpy as np
import pytz
import ta
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

exchange = ccxt.binance()
last_signals = {s: {tf: None for tf in TIMEFRAMES} for s in SYMBOLS}

# ---------------- INDICATOR FUNCTIONS ----------------
def fetch_ohlcv(symbol, timeframe, limit=LIMIT):
    logger.info(f"Fetching OHLCV for {symbol} {timeframe}...")
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
    checklist = []

    # ✅ Buy conditions
    if close > cloud_top:
        checklist.append("✅ Price ABOVE cloud")
    else:
        checklist.append("❌ Price ABOVE cloud")

    if latest["tenkan_sen"] > latest["kijun_sen"]:
        checklist.append("✅ Tenkan > Kijun")
    else:
        checklist.append("❌ Tenkan > Kijun")

    if not np.isnan(lagging):
        if lagging > cloud_top:
            checklist.append("✅ Lagging ABOVE cloud")
        elif lagging < cloud_bottom:
            checklist.append("✅ Lagging BELOW cloud")
        else:
            checklist.append("❌ Lagging inside cloud")

    # Final signal
    if close > cloud_top and latest["tenkan_sen"] > latest["kijun_sen"] and lagging > cloud_top:
        signal = "BUY"
        sl = cloud_bottom
        tp = close + 2 * (close - sl)
    elif close < cloud_bottom and latest["tenkan_sen"] < latest["kijun_sen"] and lagging < cloud_bottom:
        signal = "SELL"
        sl = cloud_top
        tp = close - 2 * (sl - close)

    return signal, close, rsi, sl, tp, checklist

# ---------------- ALERT FUNCTION -------------------
def check_and_alert(bot):
    global last_signals
    logger.info("Running scheduled market scan...")
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, tf)
                df = ichimoku(df)
                signal, price, rsi, sl, tp, checklist = analyze_ichimoku(df)
                prev_signal = last_signals[symbol][tf]

                if signal != "Neutral" and signal != prev_signal:
                    logger.info(f"🚨 New {signal} signal detected for {symbol} {tf}")
                    msg = (
                        f"🚨 {symbol} ({tf})\n"
                        f"Signal: {signal}\n"
                        f"RSI: {rsi:.1f}\n"
                        f"Entry: {price:.4f} USDT\n"
                        f"{chr(10).join(checklist)}"
                    )
                    if sl and tp:
                        msg += f"\nSL: {sl:.4f} USDT\nTP: {tp:.4f} USDT"
                    bot.send_message(chat_id=CHAT_ID, text=msg)
                    last_signals[symbol][tf] = signal
                else:
                    logger.info(f"{symbol} {tf} → No new signal ({signal})")

            except Exception as e:
                logger.error(f"Error scanning {symbol} {tf}: {e}")

# ---------------- TELEGRAM COMMANDS -------------------
def start(update, context):
    update.message.reply_text("Multi-coin Ichimoku bot online ✅")

def status(update, context):
    args = context.args
    target_symbols = SYMBOLS

    if args:
        coin = args[0].upper()
        if not coin.endswith("/USDT"):
            coin += "/USDT"
        if coin in SYMBOLS:
            target_symbols = [coin]
        else:
            update.message.reply_text(f"❌ Unsupported symbol: {args[0]}")
            return

    msg_list = []
    for symbol in target_symbols:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(symbol, tf)
                df = ichimoku(df)
                signal, price, rsi, sl, tp, checklist = analyze_ichimoku(df)
                line = f"{symbol} ({tf}): {signal} | RSI {rsi:.1f} | Price {price:.2f}"
                line += f"\n{chr(10).join(checklist)}"
                if signal in ["BUY", "SELL"]:
                    line += f"\nSL {sl:.2f} | TP {tp:.2f}"
                msg_list.append(line)
            except:
                msg_list.append(f"{symbol} ({tf}): Error fetching")
    update.message.reply_text("\n\n".join(msg_list))

def test(update, context):
    update.message.reply_text("✅ Bot is online and working!")

# ---------------- HEARTBEAT -------------------
def heartbeat(bot):
    logger.info("Sending heartbeat message...")
    bot.send_message(chat_id=CHAT_ID, text="🤖 Heartbeat: bot is alive and scanning markets.")

# ---------------- MAIN -------------------
def main():
    logger.info("Starting Ichimoku bot...")
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("test", test))

    scheduler = BackgroundScheduler(timezone=pytz.UTC)
    scheduler.add_job(check_and_alert, "interval", hours=CHECK_INTERVAL_HOURS, args=[updater.bot])
    scheduler.add_job(heartbeat, "interval", hours=4, args=[updater.bot])
    scheduler.start()

    logger.info("Bot started. Polling Telegram...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
