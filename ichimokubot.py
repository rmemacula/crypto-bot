#!/usr/bin/env python3
"""
Ichimoku + RSI Telegram Alert Bot
Features:
 - Scans coins every hour (1h, 4h, 1d)
 - Alerts when Ichimoku buy/sell forms (requires Conversion/Base, Price vs Cloud, Lagging span)
 - Includes RSI in alerts
 - Provides SL (just outside cloud) and TP (2x risk)
 - /status <COIN> shows checklist for 1h/4h/1d
 - /test checks bot alive
 - Heartbeat every 4 hours
"""

import os
import json
import math
import logging
from datetime import datetime
import time

import ccxt
import pandas as pd
import numpy as np
import pytz
from telegram import ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext
from apscheduler.schedulers.background import BackgroundScheduler
from ta.momentum import RSIIndicator

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID_ENV = os.getenv("CHAT_ID")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN environment variable.")
if not CHAT_ID_ENV:
    raise RuntimeError("Set CHAT_ID environment variable.")

try:
    CHAT_ID = int(CHAT_ID_ENV)
except Exception as e:
    raise RuntimeError("CHAT_ID must be an integer.") from e

SYMBOLS = [
    "BTC/USDT","ETH/USDT","XRP/USDT","BNB/USDT","SOL/USDT",
    "DOGE/USDT","TRX/USDT","ADA/USDT","LINK/USDT","AVAX/USDT",
    "HYPE/USDT","XLM/USDT","SUI/USDT"
]
TIMEFRAMES = ["1h","4h","1d"]
OHLCV_LIMIT = 300  # number of candles to fetch (enough for ichimoku)
CHECK_INTERVAL_HOURS = 1  # hourly scan
HEARTBEAT_INTERVAL_HOURS = 4
LAST_SIGNALS_FILE = "last_signals.json"
BUFFER_PCT = 0.0025  # small buffer for SL (0.25%)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ichimoku-bot")

# ---------------- Exchange ----------------
exchange = ccxt.binance({
    "enableRateLimit": True,
})

# ---------------- Persisted signals ----------------
def load_last_signals():
    try:
        with open(LAST_SIGNALS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # Initialize structure
        return {s: {tf: None for tf in TIMEFRAMES} for s in SYMBOLS}

def save_last_signals(state):
    try:
        with open(LAST_SIGNALS_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning("Failed to save last_signals: %s", e)

last_signals = load_last_signals()

# ---------------- Utilities & Indicators ----------------
def fetch_ohlcv_df(symbol: str, timeframe: str, limit: int = OHLCV_LIMIT) -> pd.DataFrame:
    """
    Fetch OHLCV and return DataFrame with columns time, open, high, low, close, volume
    """
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

def add_ichimoku_and_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Ichimoku lines and RSI to dataframe.
    Ichimoku:
      - tenkan (9)
      - kijun (26)
      - senkou_span_a (shift 26)
      - senkou_span_b (shift 26)
      - chikou is close shifted 26 (for checking we will use historical values)
    Also compute RSI(14) in df['rsi']
    """
    df = df.copy()
    # Ichimoku
    df["tenkan"] = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
    df["kijun"] = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
    df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
    df["senkou_b"] = ((df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2).shift(26)
    # chikou: close shifted -26 is "plotted" forward; for checks we will use comparison to close 26 bars ago
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    return df

def compute_checklist_and_signal(df: pd.DataFrame):
    """
    Returns:
      - signal: "BUY", "SELL", or "NEUTRAL"
      - rsi (float)
      - details: list strings explaining each check (for messaging)
      - entry_price
      - sl_price, tp_price (if BUY/SELL else None)
    Ichimoku rules used:
      BUY if:
        - close_now > cloud_top_now
        - tenkan > kijun (current)
        - chikou confirmation: close_now > close_26_ago AND price_26_ago > cloud_top_26  (lagging above cloud)
      SELL if symmetrical opposite (close < cloud_bottom, tenkan < kijun, chikou below cloud)
    """
    # ensure enough bars
    if len(df) < 80:
        return "NEUTRAL", None, ["Not enough data"], None, None, None

    latest = df.iloc[-1]
    # current values
    close_now = float(latest["close"])
    tenkan_now = float(latest["tenkan"])
    kijun_now = float(latest["kijun"])
    senkou_a_now = float(latest["senkou_a"]) if not np.isnan(latest["senkou_a"]) else None
    senkou_b_now = float(latest["senkou_b"]) if not np.isnan(latest["senkou_b"]) else None
    rsi_now = float(latest["rsi"]) if not np.isnan(latest["rsi"]) else None

    if senkou_a_now is None or senkou_b_now is None:
        return "NEUTRAL", rsi_now, ["Kumo not fully formed"], close_now, None, None

    cloud_top_now = max(senkou_a_now, senkou_b_now)
    cloud_bottom_now = min(senkou_a_now, senkou_b_now)

    details = []
    # Tenkan vs Kijun
    if tenkan_now > kijun_now:
        details.append("Tenkan > Kijun ✅")
    else:
        details.append("Tenkan <= Kijun ❌")

    # Price vs Cloud
    if close_now > cloud_top_now:
        details.append("Price above Cloud ✅")
    elif close_now < cloud_bottom_now:
        details.append("Price below Cloud ❌")
    else:
        details.append("Price inside Cloud ⚪")

    # Chikou checks: we need price 26 bars ago and cloud at that time
    if len(df) < 27:
        details.append("Chikou: insufficient history")
        chikou_ok = False
        chikou_cloud_ok = False
    else:
        price_26_ago = float(df["close"].iloc[-27])
        # cloud at -27 index: senkou spans are forward shifted, so their values at index -27 are what the cloud was at that time.
        sa_26 = df["senkou_a"].iloc[-27]
        sb_26 = df["senkou_b"].iloc[-27]
        if not np.isnan(sa_26) and not np.isnan(sb_26):
            cloud_top_26 = max(float(sa_26), float(sb_26))
            cloud_bottom_26 = min(float(sa_26), float(sb_26))
        else:
            cloud_top_26 = None
            cloud_bottom_26 = None

        # chikou (close now compared to close 26 ago)
        if close_now > price_26_ago:
            chikou_ok = True
            details.append("Chikou above past price ✅")
        elif close_now < price_26_ago:
            chikou_ok = False
            details.append("Chikou below past price ❌")
        else:
            chikou_ok = False
            details.append("Chikou equal to past price ⚪")

        # chikou vs cloud (use price_26_ago relative to cloud at that time)
        if cloud_top_26 is not None:
            if price_26_ago > cloud_top_26:
                chikou_cloud_ok = True
                details.append("Chikou above cloud (26 bars ago) ✅")
            elif price_26_ago < cloud_bottom_26:
                chikou_cloud_ok = False
                details.append("Chikou below cloud (26 bars ago) ❌")
            else:
                chikou_cloud_ok = False
                details.append("Chikou inside cloud (26 bars ago) ⚪")
        else:
            chikou_cloud_ok = False
            details.append("Chikou cloud data N/A")

    # Determine signal
    buy_cond = (close_now > cloud_top_now) and (tenkan_now > kijun_now) and chikou_ok and chikou_cloud_ok
    sell_cond = (close_now < cloud_bottom_now) and (tenkan_now < kijun_now) and (not chikou_ok) and (not chikou_cloud_ok)

    # Setup SL/TP when signal
    entry = close_now
    sl = None
    tp = None
    if buy_cond:
        # SL just below cloud bottom (current)
        sl = cloud_bottom_now - (BUFFER_PCT * entry)
        risk = entry - sl
        tp = entry + 2 * risk
        signal = "BUY"
    elif sell_cond:
        sl = cloud_top_now + (BUFFER_PCT * entry)
        risk = sl - entry
        tp = entry - 2 * risk
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    # Add RSI detail at end
    if rsi_now is not None:
        details.append(f"RSI: {rsi_now:.2f}")
    else:
        details.append("RSI: n/a")

    return signal, rsi_now, details, entry, sl, tp

# ---------------- Messaging ----------------
def format_alert_message(symbol, timeframe, signal, rsi, details, entry, sl, tp):
    header = f"🚨 *{symbol}* — *{signal}* ({timeframe})"
    lines = [header, f"Price: `{entry:.6f}` USDT"]
    if rsi is not None:
        lines.append(f"RSI: `{rsi:.2f}`")
    lines.append("")  # blank
    lines.append("*Ichimoku Checklist:*")
    for d in details:
        lines.append(f"- {d}")
    if sl is not None and tp is not None:
        lines.append("")
        lines.append(f"*SL:* `{sl:.6f}` USDT")
        lines.append(f"*TP:* `{tp:.6f}` USDT  (2x risk)")
        risk = abs(entry - sl)
        lines.append(f"*Risk:* `{risk:.6f}` USDT")
    return "\n".join(lines)

# ---------------- Scan & Alert ----------------
def scan_once_and_alert(context: CallbackContext):
    global last_signals
    logger.info("Running scheduled scan for symbols...")
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv_df(symbol, tf, limit=OHLCV_LIMIT)
                if df is None or df.empty:
                    logger.warning("No data for %s %s", symbol, tf)
                    continue
                df = add_ichimoku_and_rsi(df)
                signal, rsi, details, entry, sl, tp = compute_checklist_and_signal(df)
                prev = last_signals.get(symbol, {}).get(tf)
                # Only alert when new BUY/SELL forms
                if signal in ("BUY","SELL") and prev != signal:
                    msg = format_alert_message(symbol, tf, signal, rsi, details, entry, sl, tp)
                    context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)
                    logger.info("Sent alert for %s %s: %s", symbol, tf, signal)
                    # update and persist
                    if symbol not in last_signals:
                        last_signals[symbol] = {}
                    last_signals[symbol][tf] = signal
                    save_last_signals(last_signals)
                else:
                    # update stored state for neutrality -> don't overwrite buy/sell unless changes
                    if symbol not in last_signals:
                        last_signals[symbol] = {}
                        last_signals[symbol][tf] = signal
                # small delay to avoid hitting rate limit too hard
                time.sleep(0.5)
            except Exception as e:
                logger.exception("Error scanning %s %s: %s", symbol, tf, e)

# ---------------- Heartbeat ----------------
def heartbeat(context: CallbackContext):
    """Send a heartbeat every n hours so you know the bot is alive."""
    ts = datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        context.bot.send_message(chat_id=CHAT_ID, text=f"💓 Heartbeat: Bot alive at {ts}")
        logger.info("Heartbeat sent.")
    except Exception as e:
        logger.exception("Failed to send heartbeat: %s", e)

# ---------------- /status command ----------------
def status_command(update, context):
    """
    /status BTC  -> show 1h/4h/1d checklist for BTC
    also supports passing multiple coins: /status BTC ETH SOL
    """
    args = context.args
    if not args:
        update.message.reply_text("Usage: /status BTC (or multiple like /status BTC ETH)")
        return

    messages = []
    for coin_token in args:
        coin = coin_token.upper()
        symbol = f"{coin}/USDT"
        if symbol not in SYMBOLS:
            messages.append(f"{coin}: not tracked.")
            continue

        summary_blocks = []
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv_df(symbol, tf, limit=OHLCV_LIMIT)
                if df is None or df.empty:
                    summary_blocks.append(f"{tf}: no data")
                    continue
                df = add_ichimoku_and_rsi(df)
                signal, rsi, details, entry, sl, tp = compute_checklist_and_signal(df)
                # Present checklist
                block = [f"*{symbol}* — _{tf}_ → *{signal}*"]
                for d in details:
                    # Add emoji already present in details
                    block.append(f"- {d}")
                if entry is not None:
                    block.append(f"Price: `{entry:.6f}` USDT")
                if sl is not None and tp is not None:
                    block.append(f"SL: `{sl:.6f}`  TP: `{tp:.6f}`")
                summary_blocks.append("\n".join(block))
            except Exception as e:
                logger.exception("Error building status for %s %s", symbol, tf)
                summary_blocks.append(f"{tf}: error")
        messages.append("\n\n".join(summary_blocks))
    # Telegram message size: keep it reasonable
    final = "\n\n---\n\n".join(messages)
    update.message.reply_text(final, parse_mode=ParseMode.MARKDOWN)

# ---------------- /test command ----------------
def test_command(update, context):
    update.message.reply_text("✅ Bot is running and connected.")

# ---------------- Main & Scheduler ----------------
def main():
    logger.info("Starting Ichimoku bot...")
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    # Commands
    dp.add_handler(CommandHandler("test", test_command))
    dp.add_handler(CommandHandler("status", status_command))

    # Scheduler (pytz timezone)
    scheduler = BackgroundScheduler(timezone=pytz.UTC)
    # hourly scans
    scheduler.add_job(scan_once_and_alert, "interval", hours=CHECK_INTERVAL_HOURS, args=[updater.job_queue])
    # heartbeat every 4 hours
    scheduler.add_job(heartbeat, "interval", hours=HEARTBEAT_INTERVAL_HOURS, args=[updater.job_queue])
    scheduler.start()

    # also run an immediate scan at startup
    try:
        # call job directly with the bot context
        scan_once_and_alert(updater.bot)
    except Exception as e:
        logger.exception("Initial scan failed: %s", e)

    updater.start_polling()
    logger.info("Bot started, polling Telegram.")
    updater.idle()
    scheduler.shutdown()

if __name__ == "__main__":
    main()
