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
    "HYPEUSDT", "XLMUSDT", "SUIUSDT",
    # plus more:
    "MATICUSDT", "LTCUSDT", "DOTUSDT", "NEARUSDT", "ATOMUSDT",
    "UNIUSDT", "FILUSDT", "ICPUSDT", "APEUSDT", "MKRUSDT",
    "EOSUSDT", "AAVEUSDT", "SNXUSDT", "KSMUSDT", "ENJUSDT",
    "GRTUSDT", "SANDUSDT", "AXSUSDT", "ICPUSDT", "ALGOUSDT",
    "FTMUSDT", "VETUSDT", "THETAUSDT", "CHZUSDT", "XMRUSDT",
    "ZECUSDT", "EOSUSDT", "FLOWUSDT", "KLAYUSDT", "MANAUSDT",
    "QNTUSDT", "CRVUSDT", "CELOUSDT", "KAVAUSDT", "NEOUSDT",
    "RPLUSDT", "LDOUSDT", "RUNEUSDT", "CHSBUSDT",
    # etc (ensure 50 total non-stable symbols)
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
        if r.status_code != 200:
            logging.warning(f"Fetch fail {symbol} {interval}: HTTP {r.status_code}")
            return None
        data = r.json()
        if isinstance(data, dict) and data.get("code"):
            logging.warning(f"Binance API error for {symbol} {interval}: {data}")
            return None
        df = pd.DataFrame(data, columns=[
            "time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"
        ])
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Fetch exception {symbol} {interval}: {e}")
        return None

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    if df is None or len(df) < 104:  # Need 52 + 26 + 26 for proper analysis
        return {"signal": "Neutral", "price": None, "rsi": None}

    close = df["c"]
    high = df["h"]
    low = df["l"]

    # ---- Ichimoku Components ----
    # Tenkan-sen (Conversion Line): 9-period
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (Base Line): 26-period
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A): Average of Tenkan and Kijun
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): 52-period
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # ---- RSI ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-2]) if not np.isnan(rsi.iloc[-2]) else None

    # ---- Last closed candle analysis ----
    last_idx = -2
    price = close.iloc[last_idx]
    tenkan_v = float(tenkan.iloc[last_idx])
    kijun_v = float(kijun.iloc[last_idx])
    
    # ---- Current Cloud Position (where price is NOW) ----
    # The cloud at the current candle was projected 26 periods ago
    # So we need to look back 26 periods in the Senkou calculations
    cloud_idx = last_idx - 26
    
    # Make sure we have enough historical data
    if abs(cloud_idx) <= len(df) - 52:
        cloud_a_current = float(senkou_a.iloc[cloud_idx])
        cloud_b_current = float(senkou_b.iloc[cloud_idx])
    else:
        # Fallback if not enough data
        cloud_a_current = float(senkou_a.iloc[last_idx])
        cloud_b_current = float(senkou_b.iloc[last_idx])

    # ---- Chikou Span (Lagging Span) Analysis ----
    # Chikou Span = Current close plotted 26 periods back
    # We compare current price to the cloud at the chikou position (26 periods ago)
    chikou_above = chikou_below = False
    
    # The chikou position is 26 periods back from current
    chikou_idx = last_idx - 26
    
    # The cloud at the chikou position was projected 26 periods before that
    # So we look back another 26 periods in the Senkou calculations
    chikou_cloud_idx = chikou_idx - 26
    
    # Make sure we have enough data
    if abs(chikou_cloud_idx) <= len(df) - 52:
        # Cloud values at the chikou position (what the cloud was 26 periods ago)
        cloud_a_at_chikou = float(senkou_a.iloc[chikou_cloud_idx])
        cloud_b_at_chikou = float(senkou_b.iloc[chikou_cloud_idx])
        
        # Current price (chikou span value) compared to the cloud at that historical position
        if not np.isnan(cloud_a_at_chikou) and not np.isnan(cloud_b_at_chikou):
            chikou_above = price > max(cloud_a_at_chikou, cloud_b_at_chikou)
            chikou_below = price < min(cloud_a_at_chikou, cloud_b_at_chikou)

    # ---- Future Cloud Analysis (26 periods ahead projection) ----
    # The Senkou values calculated NOW represent where the cloud WILL BE 26 periods ahead
    future_cloud_bullish = future_cloud_bearish = False
    
    cloud_a_future = float(senkou_a.iloc[last_idx])
    cloud_b_future = float(senkou_b.iloc[last_idx])
    
    if not np.isnan(cloud_a_future) and not np.isnan(cloud_b_future):
        # Future cloud is bullish when Senkou A > Senkou B (green/bullish cloud ahead)
        # Future cloud is bearish when Senkou A < Senkou B (red/bearish cloud ahead)
        future_cloud_bullish = cloud_a_future > cloud_b_future
        future_cloud_bearish = cloud_a_future < cloud_b_future

    # ---- Ichimoku Checklist ----
    checklist_bull = [
        ("Price above cloud", price > max(cloud_a_current, cloud_b_current)),
        ("Tenkan > Kijun", tenkan_v > kijun_v),
        ("Chikou above cloud", chikou_above),
        ("Future cloud bullish", future_cloud_bullish),
    ]

    checklist_bear = [
        ("Price below cloud", price < min(cloud_a_current, cloud_b_current)),
        ("Tenkan < Kijun", tenkan_v < kijun_v),
        ("Chikou below cloud", chikou_below),
        ("Future cloud bearish", future_cloud_bearish),
    ]

    bullish_count = sum(c for _, c in checklist_bull)
    bearish_count = sum(c for _, c in checklist_bear)

    # ---- Signal Generation and Risk Management ----
    signal = "Neutral"
    sl = tp = None
    
    if bullish_count >= 3:
        signal = "BUY"
        # Stop loss below the current cloud
        sl = min(cloud_a_current, cloud_b_current) * 0.995
        # Take profit at 2:1 reward-risk ratio
        tp = price + 2 * (price - sl)
    elif bearish_count >= 3:
        signal = "SELL"
        # Stop loss above the current cloud
        sl = max(cloud_a_current, cloud_b_current) * 1.005
        # Take profit at 2:1 reward-risk ratio
        tp = price - 2 * (sl - price)

    # Validate SL and TP
    if sl is None or tp is None or np.isnan(sl) or np.isnan(tp):
        sl = tp = None

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
        "tenkan": tenkan_v,
        "kijun": kijun_v,
        "cloud_a": cloud_a_current,
        "cloud_b": cloud_b_current,
        "cloud_a_future": cloud_a_future,
        "cloud_b_future": cloud_b_future,
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
            messages.append(f"❌ No data for {tf_label} (check symbol or API)")
            continue
        if len(df) < 104:
            messages.append(f"❌ Not enough data for {tf_label} (need 104+ candles, have {len(df)})")
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

        msg += "\n" + format_checklist(analysis)
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
                bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                last_signals[k] = sent_label
                save_last_signals(last_signals)

#---------------status1d--------------------
from datetime import datetime, timezone, timedelta

def status1d(update: Update, context: CallbackContext):
    tf_label = "1d"
    interval = TIMEFRAMES[tf_label]

    update.message.reply_text("⏳ Scanning 1D Ichimoku + RSI signals (4/4 confirmed only)...")

    buy_msgs, sell_msgs = [], []
    manila_tz = timezone(timedelta(hours=8))  # UTC+8 timezone

    for sym in SYMBOLS:
        df = fetch_ohlcv(sym, interval)
        if df is None or len(df) < 104:
            continue

        analysis = analyze_df(df)
        signal = analysis["signal"]

        # Get timestamp of last closed candle (convert milliseconds to datetime)
        last_candle_time = int(df.iloc[-2]["time"])  # Binance gives ms timestamp
        ts = datetime.fromtimestamp(last_candle_time / 1000, tz=manila_tz).strftime("%Y-%m-%d %H:%M %p (Manila)")

        # Only include if all 4 checklist items are met for that side
        if signal == "BUY" and analysis["bull_count"] == 4:
            msg = (
                f"🟩 *{sym}* — STRONG BUY (4/4)\n"
                f"🕒 Time: {ts}\n"
                f"💰 Price: {analysis['price']:.2f} USDT\n"
                f"📊 RSI: {analysis['rsi']:.2f}\n"
                f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n"
            )
            if analysis["sl"] and analysis["tp"]:
                msg += f"🎯 SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"
            msg += "\n" + format_checklist(analysis)
            buy_msgs.append(msg)

        elif signal == "SELL" and analysis["bear_count"] == 4:
            msg = (
                f"🟥 *{sym}* — STRONG SELL (4/4)\n"
                f"🕒 Time: {ts}\n"
                f"💰 Price: {analysis['price']:.2f} USDT\n"
                f"📊 RSI: {analysis['rsi']:.2f}\n"
                f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n"
            )
            if analysis["sl"] and analysis["tp"]:
                msg += f"🎯 SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"
            msg += "\n" + format_checklist(analysis)
            sell_msgs.append(msg)

    # Send grouped results
    if buy_msgs:
        update.message.reply_text(
            "🟩 *STRONG BUY signals (4/4 confirmed)*\n\n" + "\n\n".join(buy_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if sell_msgs:
        update.message.reply_text(
            "🟥 *STRONG SELL signals (4/4 confirmed)*\n\n" + "\n\n".join(sell_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if not buy_msgs and not sell_msgs:
        update.message.reply_text("⚪ No coins met all 4 Ichimoku checklist conditions (1D).")

    update.message.reply_text("✅ 1D scan complete.")

# ---------------- HEARTBEAT ----------------
def heartbeat(context: CallbackContext):
    context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("test", test))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("status1d", status1d))
    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=300, first=10)
    jq.run_repeating(heartbeat, interval=14400, first=20)
    logging.info("Bot started")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()