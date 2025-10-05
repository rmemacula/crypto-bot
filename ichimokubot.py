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

# ---------------- TRADINGVIEW LINK ----------------
def tradingview_link(symbol, tf_label):
    tf_map = {"1h": "60", "4h": "240", "1d": "1D"}
    interval = tf_map.get(tf_label, "60")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={interval}"

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
    # ---- Ichimoku Calculations ----
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
    price = close.iloc[-2]  # last closed candle

    # ---- RSI Calculation ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-2]

    # ---- Use last closed candle values ----
    tenkan_v = tenkan.iloc[-2]
    kijun_v = kijun.iloc[-2]

    # ---- Future Cloud (projected 26 candles ahead) ----
    span_a_v = senkou_span_a.dropna().iloc[-1]
    span_b_v = senkou_span_b.dropna().iloc[-1]

    # ---- Lagging span (Chikou) ----
    chikou_span = close.shift(-26)
    chikou_above = chikou_below = False
    if len(df) > 26:
        idx = -28
        past_close = close.iloc[idx]
        past_span_a = senkou_span_a.iloc[idx] if not np.isnan(senkou_span_a.iloc[idx]) else 0
        past_span_b = senkou_span_b.iloc[idx] if not np.isnan(senkou_span_b.iloc[idx]) else 0
        chikou_above = past_close > max(past_span_a, past_span_b)
        chikou_below = past_close < min(past_span_a, past_span_b)

    # ✅ Ichimoku Checklist
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
    signal = analysis["signal"]

    for (bull_label, bull_val), (bear_label, bear_val) in zip(
        analysis["checklist_bull"], analysis["checklist_bear"]
    ):
        if signal == "BUY":
            lines.append(f"{'✅' if bull_val else '❌'} {bull_label}")
        elif signal == "SELL":
            lines.append(f"{'✅' if bear_val else '❌'} {bear_label}")
        else:
            if bull_val:
                lines.append("✅ " + bull_label)
            elif bear_val:
                lines.append("✅ " + bear_label)
            else:
                lines.append("❌ " + bull_label + " / " + bear_label)
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
            f"📈 [View on TradingView]({tradingview_link(sym, tf_label)})\n"
        )
        if analysis["sl"] and analysis["tp"]:
            msg += f"SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"

        msg += format_checklist(analysis)
        messages.append(msg)

    update.message.reply_text("\n\n".join(messages), parse_mode="Markdown")

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
                # Build TradingView link safely
                tv_link = tradingview_link(symbol, tf_label)
                safe_symbol = symbol.replace("_", "\\_").replace("-", "\\-")

                msg = (
                    f"🚨 *{safe_symbol}* ({tf_label}) — *{sig}*\n\n"
                    f"💰 *Price:* {analysis['price']:.2f} USDT\n"
                    f"📊 *RSI:* {analysis['rsi']:.2f}\n"
                    f"🔗 [View on TradingView]({tv_link})\n\n"
                )

                if analysis["sl"] and analysis["tp"]:
                    msg += f"🎯 *SL:* {analysis['sl']:.2f} | *TP:* {analysis['tp']:.2f}\n\n"

                msg += format_checklist(analysis)

                # Send formatted message
                bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

                # Update last signal
                last_signals[k] = sent_label
                save_last_signals(last_signals)
# ---------------- HEARTBEAT ----------------
def heartbeat(context: CallbackContext):
    context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    updater.bot.defaults.parse_mode = "Markdown"  # ✅ Makes links clickable
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("test", test))
    dp.add_handler(CommandHandler("status", status))

    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=300, first=10)
    jq.run_repeating(heartbeat, interval=14400, first=20)

    logging.info("Bot started")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
