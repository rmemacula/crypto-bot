import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
    "HYPEUSDT", "XLMUSDT", "SUIUSDT"
]

TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d"}

DATA_FILE = "last_signals.json"

logging.basicConfig(level=logging.INFO)

# ---------------- STORAGE ----------------
def load_last_signals():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_last_signals(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

last_signals = load_last_signals()

def key_for(symbol, tf):
    return f"{symbol}_{tf}"

# ---------------- DATA FETCH ----------------
def fetch_ohlcv(symbol, interval, limit=150):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if isinstance(data, dict) and data.get("code"):
            return None
        df = pd.DataFrame(data, columns=[
            "time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"
        ])
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Fetch fail {symbol} {interval}: {e}")
        return None

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    high9 = df["h"].rolling(window=9).max()
    low9 = df["l"].rolling(window=9).min()
    tenkan = (high9 + low9) / 2

    high26 = df["h"].rolling(window=26).max()
    low26 = df["l"].rolling(window=26).min()
    kijun = (high26 + low26) / 2

    senkou_span_a = ((tenkan + kijun) / 2).shift(26)
    high52 = df["h"].rolling(window=52).max()
    low52 = df["l"].rolling(window=52).min()
    senkou_span_b = ((high52 + low52) / 2).shift(26)

    close = df["c"]
    price = close.iloc[-1]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]

    tenkan_v = tenkan.iloc[-2]
    kijun_v = kijun.iloc[-2]
    span_a_v = senkou_span_a.iloc[-2]
    span_b_v = senkou_span_b.iloc[-2]
    price = df["c"].iloc[-2]

    # ---------------- CORRECT LAGGING SPAN CHECK ----------------
    chikou_above = False
    chikou_below = False
    if len(df) > 26:
        idx = -27  # 26 bars ago
        past_close = close.iloc[idx]
        span_a_past = senkou_span_a.iloc[idx]
        span_b_past = senkou_span_b.iloc[idx]

        if not np.isnan(span_a_past) and not np.isnan(span_b_past):
            chikou_above = past_close > max(span_a_past, span_b_past)
            chikou_below = past_close < min(span_a_past, span_b_past)

    # ✅ Ichimoku checklist (aligned)
    checklist_bull = [
        ("Price above cloud", price > max(span_a_v, span_b_v)),
        ("Tenkan > Kijun", tenkan_v > kijun_v),
        ("Lagging span above cloud", chikou_above),
        ("Future cloud bullish", span_a_v > span_b_v),
    ]

    checklist_bear = [
        ("Price below cloud", price < min(span_a_v, span_b_v)),
        ("Tenkan < Kijun", tenkan_v < kijun_v),
        ("Lagging span below cloud", chikou_below),
        ("Future cloud bearish", span_a_v < span_b_v),
    ]

    bullish_count = sum(c for _, c in checklist_bull)
    bearish_count = sum(c for _, c in checklist_bear)

    signal = "Neutral"
    sl = tp = None
    if bullish_count >= 3:
        signal = "BUY"
        sl = min(span_a_v, span_b_v) * 0.995
        risk = price - sl
        tp = price + 2 * risk
    elif bearish_count >= 3:
        signal = "SELL"
        sl = max(span_a_v, span_b_v) * 1.005
        risk = sl - price
        tp = price - 2 * risk

    return {
        "price": price,
        "rsi": rsi_val,
        "signal": signal,
        "bull_count": bullish_count,
        "bear_count": bearish_count,
        "checklist_bull": checklist_bull,
        "checklist_bear": checklist_bear,
        "sl": sl,
        "tp": tp,
    }

# ---------------- CHECKLIST FORMATTER ----------------
def format_checklist(analysis):
    lines = []
    for bull, bear in zip(analysis["checklist_bull"], analysis["checklist_bear"]):
        if bull[1]:
            lines.append("✅ " + bull[0])
        elif bear[1]:
            lines.append("✅ " + bear[0])
        else:
            lines.append("❌ " + bull[0] + " / " + bear[0])
    return "\n".join(lines)

# ---------------- COMMANDS ----------------
def test(update: Update, context: CallbackContext):
    update.message.reply_text("✅ Bot is working!")

def status(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Usage: /status BTC")
        return
    sym = context.args[0].upper() + "USDT"
    if sym not in SYMBOLS:
        update.message.reply_text("Unknown coin")
        return

    messages = []
    for tf_label, interval in TIMEFRAMES.items():
        df = fetch_ohlcv(sym, interval)
        if df is None:
            messages.append(f"❌ No data for {tf_label}")
            continue
        analysis = analyze_df(df)

        msg = (
            f"📊 {sym} ({tf_label})\n"
            f"Signal: {analysis['signal']}\n"
            f"Price: {analysis['price']:.2f} USDT\n"
            f"RSI: {analysis['rsi']:.2f}\n"
        )
        if analysis["sl"] and analysis["tp"]:
            msg += f"SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"

        msg += format_checklist(analysis)
        messages.append(msg)

    update.message.reply_text("\n\n".join(messages))

# ---------------- ALERT JOB ----------------
def check_and_alert(context: CallbackContext):
    global last_signals
    bot = context.bot
    for symbol in SYMBOLS:
        for tf_label, interval in TIMEFRAMES.items():
            df = fetch_ohlcv(symbol, interval)
            if df is None:
                continue
            analysis = analyze_df(df)
            sig = analysis["signal"]
            k = key_for(symbol, tf_label)
            prev = last_signals.get(k)

            sent_label = f"{sig}|{analysis['bull_count']}|{analysis['bear_count']}"
            if sig in ("BUY", "SELL") and prev != sent_label:
                msg = (
                    f"🚨 {symbol} ({tf_label}) — {sig}\n\n"
                    f"Price: {analysis['price']:.2f} USDT | RSI: {analysis['rsi']:.2f}\n"
                )
                if analysis["sl"] and analysis["tp"]:
                    msg += f"SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n\n"
                msg += format_checklist(analysis)

                bot.send_message(chat_id=CHAT_ID, text=msg)
                last_signals[k] = sent_label
                save_last_signals(last_signals)

# ---------------- HEARTBEAT ----------------
def heartbeat(context: CallbackContext):
    context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("test", test))
    dp.add_handler(CommandHandler("status", status))

    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=300, first=10)    # every 5 mins
    jq.run_repeating(heartbeat, interval=14400, first=20)        # every 4 hours

    logging.info("Bot started")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
