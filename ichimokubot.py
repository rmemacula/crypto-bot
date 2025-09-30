# ---------------- alert job ----------------
def check_and_alert(bot):
    global last_signals
    logger.info("Scheduled scan started: checking symbols/timeframes...")
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                logger.info("Scanning %s %s", symbol, tf)
                df = fetch_ohlcv_df(symbol, tf)
                if df is None or df.empty:
                    logger.warning("No data for %s %s", symbol, tf)
                    continue

                signal, price, rsi, checklist, sl, tp, strength_count, strength_label = analyze_df_for_signal(df)
                prev = last_signals.get(symbol, {}).get(tf)
                logger.info("%s %s -> %s (strength=%s, prev=%s)", symbol, tf, signal if signal != "Neutral" else strength_label, strength_count, prev)

                # --- alert rules ---
                alert_needed = False
                if signal in ("BUY", "SELL") and strength_count == 4:
                    alert_needed = (prev != signal)
                elif (strength_count == 3):  # NEW weaker threshold alert
                    # Encode BUY/SELL depending on which side won
                    if "Buy" in strength_label and prev != "Buy(3/4)":
                        alert_needed = True
                    elif "Sell" in strength_label and prev != "Sell(3/4)":
                        alert_needed = True

                if alert_needed:
                    header = f"🚨 *{symbol}* ({tf}) — *{signal if signal!='Neutral' else ''}* ({strength_label})"
                    body = f"Price: `{price:.6f}` USDT  |  RSI: `{rsi:.2f}`\n\n"
                    checklist_text = format_checklist_text(checklist)
                    sltp = ""
                    # Add SL/TP only for 4/4 confirmed signals
                    if strength_count == 4 and sl is not None and tp is not None:
                        sltp = f"\n\n*SL:* `{sl:.6f}` USDT\n*TP:* `{tp:.6f}` USDT  (2x risk)"
                    msg = header + "\n\n" + body + checklist_text + sltp

                    logger.info("ALERT -> %s %s: %s", symbol, tf, strength_label)
                    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)

                    # persist last signal
                    if symbol not in last_signals:
                        last_signals[symbol] = {}
                    last_signals[symbol][tf] = strength_label
                    save_last_signals(last_signals)

                else:
                    logger.debug("No new alert for %s %s (signal=%s, strength=%s)", symbol, tf, signal, strength_count)

                time.sleep(0.25)
            except Exception as e:
                logger.exception("Error scanning %s %s: %s", symbol, tf, e)
