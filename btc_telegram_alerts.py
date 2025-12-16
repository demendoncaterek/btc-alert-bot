import asyncio
import json
import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import requests

# =========================
# CONFIG (env override)
# =========================
PRODUCT = os.getenv("PRODUCT", "BTC-USD")
GRANULARITY = int(os.getenv("GRANULARITY", "60"))  # 60s candles
STATE_FILE = os.getenv("STATE_FILE", "btc_state.json")

# Trading state for UI + paper trading
TRADE_FILE = os.getenv("TRADE_FILE", "btc_trades.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")  # string is fine

COINBASE_BASE = "https://api.exchange.coinbase.com"

ENGINE_LOOP_SECS = int(os.getenv("ENGINE_LOOP_SECS", "5"))

ALERT_COOLDOWN = int(os.getenv("ALERT_COOLDOWN", "300"))
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "70"))

# Heartbeat (optional): 0 = off (recommended). Set e.g. 3600 for hourly.
HEARTBEAT_SECS = int(os.getenv("HEARTBEAT_SECS", "0"))

# Paper trading defaults (SAFE)
TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()  # "paper" (default), "live" (not implemented)
STARTING_CASH = float(os.getenv("STARTING_CASH", "1000"))
TRADE_FRACTION = float(os.getenv("TRADE_FRACTION", "1.0"))  # 1.0 = use all cash on buy

# =========================
# INTERNAL STATE
# =========================
LATEST_STATE: Dict[str, Any] = {}
LAST_ALERT_TS = 0.0
LAST_HEARTBEAT_TS = 0.0

TRADE_STATE: Dict[str, Any] = {
    "enabled": False,
    "mode": TRADE_MODE,          # paper/live
    "cash_usd": STARTING_CASH,
    "btc_qty": 0.0,
    "entry_price": 0.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "equity_usd": STARTING_CASH,
    "last_trade": None,
    "trades": [],               # list of trades for UI markers
    "updated_at": None,
}

# Telegram polling offset (kept in memory; fine for most cases)
TG_OFFSET = 0

# =========================
# HELPERS
# =========================
def atomic_write_json(path: str, obj: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def safe_load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
            if not raw:
                return None
            return json.loads(raw)
    except:
        return None


def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=8,
        )
    except:
        pass


def fetch_candles(limit=60):
    url = f"{COINBASE_BASE}/products/{PRODUCT}/candles"
    resp = requests.get(
        url,
        params={"granularity": GRANULARITY},
        headers={"Accept": "application/json", "User-Agent": "btc-alerts"},
        timeout=10,
    )
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Coinbase API error: {data}")

    data.sort(key=lambda x: x[0])
    data = data[-limit:]

    candles = []
    closes = []

    for c in data:
        ts = int(c[0])
        dt = datetime.fromtimestamp(ts)
        candles.append(
            {
                "ts": ts,
                "time": dt.strftime("%H:%M"),
                "open": float(c[3]),
                "high": float(c[2]),
                "low": float(c[1]),
                "close": float(c[4]),
            }
        )
        closes.append(float(c[4]))

    return candles, closes


def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0

    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = -np.minimum(deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def confidence_score(rsi, trend_strength, momentum):
    score = 0

    # RSI extremes help; mid-range hurts
    if rsi < 30 or rsi > 70:
        score += 35
    elif 35 <= rsi <= 65:
        score -= 10

    # momentum + trend strength scale-in
    score += min(abs(momentum) * 2000, 25)
    score += min(trend_strength * 2000, 25)

    return max(0, min(100, int(score)))


def trade_equity(price: float) -> float:
    return float(TRADE_STATE["cash_usd"]) + float(TRADE_STATE["btc_qty"]) * price


def record_trade(side: str, price: float, qty: float, pnl: float):
    t = {
        "ts": int(time.time()),
        "time": datetime.now().strftime("%H:%M:%S"),
        "side": side,  # BUY / SELL
        "price": round(price, 2),
        "qty": qty,
        "pnl": round(pnl, 2),
    }
    TRADE_STATE["trades"].append(t)
    # keep last 200
    TRADE_STATE["trades"] = TRADE_STATE["trades"][-200:]
    TRADE_STATE["last_trade"] = t


def maybe_paper_trade(signal: str, price: float, confidence: int):
    """
    Paper trading:
    - BUY: enter long using TRADE_FRACTION of cash if flat
    - SELL: exit full position if long
    Only runs if /starttrade enabled AND confidence >= MIN_CONFIDENCE
    """
    if not TRADE_STATE["enabled"]:
        return
    if confidence < MIN_CONFIDENCE:
        return

    cash = float(TRADE_STATE["cash_usd"])
    btc = float(TRADE_STATE["btc_qty"])
    entry = float(TRADE_STATE["entry_price"])

    if signal == "BUY" and btc <= 0:
        spend = cash * max(0.0, min(TRADE_FRACTION, 1.0))
        if spend <= 1:
            return
        qty = spend / price
        TRADE_STATE["cash_usd"] = cash - spend
        TRADE_STATE["btc_qty"] = qty
        TRADE_STATE["entry_price"] = price
        record_trade("BUY", price, qty, 0.0)
        send_telegram(
            f"🟢 PAPER BUY\nPrice: ${price:,.2f}\nQty: {qty:.8f} BTC\nConfidence: {confidence}%"
        )

    elif signal == "SELL" and btc > 0:
        proceeds = btc * price
        cost = btc * entry
        pnl = proceeds - cost
        TRADE_STATE["cash_usd"] = cash + proceeds
        TRADE_STATE["btc_qty"] = 0.0
        TRADE_STATE["entry_price"] = 0.0
        TRADE_STATE["realized_pnl"] = float(TRADE_STATE["realized_pnl"]) + pnl
        record_trade("SELL", price, btc, pnl)
        send_telegram(
            f"🔴 PAPER SELL\nPrice: ${price:,.2f}\nPnL: ${pnl:,.2f}\nConfidence: {confidence}%"
        )


def update_trade_metrics(price: float):
    btc = float(TRADE_STATE["btc_qty"])
    entry = float(TRADE_STATE["entry_price"])
    unreal = 0.0
    if btc > 0 and entry > 0:
        unreal = btc * (price - entry)

    TRADE_STATE["unrealized_pnl"] = unreal
    TRADE_STATE["equity_usd"] = trade_equity(price)
    TRADE_STATE["updated_at"] = datetime.now().strftime("%H:%M:%S")


def persist_trade_state():
    atomic_write_json(TRADE_FILE, TRADE_STATE)


# =========================
# TELEGRAM COMMANDS (polling)
# =========================
def telegram_get_updates(timeout_s=25):
    global TG_OFFSET
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": timeout_s, "offset": TG_OFFSET + 1}
    r = requests.get(url, params=params, timeout=timeout_s + 5)
    return r.json()


def handle_command(text: str, chat_id: str):
    text = (text or "").strip().lower()

    if text in ("/help", "help"):
        send_telegram(
            "Commands:\n"
            "/status - show engine + trading status\n"
            "/starttrade - enable paper trading\n"
            "/stoptrade - disable paper trading\n"
        )
        return

    if text.startswith("/status"):
        s = LATEST_STATE or {}
        price = s.get("price", 0)
        rsi = s.get("rsi", 0)
        trend = s.get("trend", "WAIT")
        conf = s.get("confidence", 0)
        updated = s.get("time", "--:--:--")
        enabled = "ON ✅" if TRADE_STATE["enabled"] else "OFF ⛔"
        pos = "LONG" if float(TRADE_STATE["btc_qty"]) > 0 else "FLAT"
        eq = float(TRADE_STATE.get("equity_usd", 0))
        rpnl = float(TRADE_STATE.get("realized_pnl", 0))
        upnl = float(TRADE_STATE.get("unrealized_pnl", 0))

        send_telegram(
            "📊 BTC STATUS\n"
            f"Price: ${price:,.2f}\n"
            f"RSI(1m): {rsi}\n"
            f"Trend: {trend}\n"
            f"Confidence: {conf}%\n"
            f"Updated: {updated}\n\n"
            f"Trading: {enabled} ({TRADE_STATE['mode']})\n"
            f"Position: {pos}\n"
            f"Equity: ${eq:,.2f}\n"
            f"Realized PnL: ${rpnl:,.2f}\n"
            f"Unrealized PnL: ${upnl:,.2f}"
        )
        return

    if text.startswith("/starttrade"):
        TRADE_STATE["enabled"] = True
        persist_trade_state()
        send_telegram("✅ Trading enabled (PAPER). I will only execute when Confidence >= MIN_CONFIDENCE.")
        return

    if text.startswith("/stoptrade"):
        TRADE_STATE["enabled"] = False
        persist_trade_state()
        send_telegram("🛑 Trading disabled.")
        return

    # ignore other messages


def telegram_poll_worker():
    global TG_OFFSET
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    # Only respond to your configured chat id
    allowed_chat = str(TELEGRAM_CHAT_ID)

    while True:
        try:
            data = telegram_get_updates(timeout_s=25)
            if not data.get("ok"):
                time.sleep(3)
                continue

            for upd in data.get("result", []):
                TG_OFFSET = max(TG_OFFSET, upd.get("update_id", 0))
                msg = upd.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = str(chat.get("id", ""))

                if chat_id != allowed_chat:
                    continue

                text = msg.get("text", "")
                if text:
                    handle_command(text, chat_id)

        except Exception:
            time.sleep(3)


# =========================
# STARTUP MESSAGE (only once per container)
# =========================
def send_startup_message_once():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    flag = "/tmp/btc_engine_startup_sent.flag"
    if os.path.exists(flag):
        return
    send_telegram("✅ BTC AI bot is live and can send alerts.\nTip: send /status anytime.")
    try:
        with open(flag, "w") as f:
            f.write("1")
    except:
        pass


# =========================
# MAIN LOOP
# =========================
async def main():
    global LATEST_STATE, LAST_ALERT_TS, LAST_HEARTBEAT_TS

    # load existing trade state if present (keeps your paper portfolio across restarts)
    loaded_trade = safe_load_json(TRADE_FILE)
    if isinstance(loaded_trade, dict):
        TRADE_STATE.update(loaded_trade)

    send_startup_message_once()

    # start telegram polling thread
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        t = threading.Thread(target=telegram_poll_worker, daemon=True)
        t.start()

    print("✅ BTC Alert Engine Running (AI-Filtered • Short-Term)")

    while True:
        try:
            candles, closes = fetch_candles(limit=60)
            price = closes[-1]
            rsi = compute_rsi(closes)

            momentum = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 6 else 0.0
            trend_strength = abs(momentum)

            trend = "WAIT"
            signal = "WAIT"

            # simple rules (your current logic)
            if rsi < 30 and momentum > 0:
                trend = signal = "BUY"
            elif rsi > 70 and momentum < 0:
                trend = signal = "SELL"

            confidence = confidence_score(rsi, trend_strength, momentum)

            state_obj = {
                "price": round(price, 2),
                "rsi": round(rsi, 1),
                "trend": trend,
                "state": signal,
                "confidence": confidence,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": candles[-30:],
                "momentum": float(momentum),
                "notes": f"src=Coinbase • momentum={momentum:.5f}",
                "error": "",
            }
            LATEST_STATE = state_obj
            atomic_write_json(STATE_FILE, state_obj)

            # Paper trade if enabled
            if TRADE_STATE["mode"] == "paper":
                maybe_paper_trade(signal, price, confidence)
            else:
                # live trading not implemented on purpose (safety)
                pass

            update_trade_metrics(price)
            persist_trade_state()

            # Telegram alerts (signals)
            now = time.time()
            if (
                signal in ("BUY", "SELL")
                and confidence >= MIN_CONFIDENCE
                and now - LAST_ALERT_TS > ALERT_COOLDOWN
            ):
                send_telegram(
                    f"📢 BTC {signal} ALERT\n"
                    f"Price: ${price:,.2f}\n"
                    f"RSI(1m): {round(rsi,1)}\n"
                    f"Confidence: {confidence}%"
                )
                LAST_ALERT_TS = now

            # Optional heartbeat (avoid spam; off by default)
            if HEARTBEAT_SECS > 0 and now - LAST_HEARTBEAT_TS > HEARTBEAT_SECS:
                send_telegram(
                    f"💓 BTC AI heartbeat\nPrice: ${price:,.2f}\nRSI(1m): {round(rsi,1)}\nTrend: {trend}"
                )
                LAST_HEARTBEAT_TS = now

        except Exception as e:
            err_obj = {
                "price": 0,
                "rsi": 0,
                "trend": "WAIT",
                "state": "WAIT",
                "confidence": 0,
                "time": datetime.now().strftime("%H:%M:%S"),
                "candles": [],
                "momentum": 0.0,
                "notes": "",
                "error": str(e),
            }
            LATEST_STATE = err_obj
            atomic_write_json(STATE_FILE, err_obj)

        await asyncio.sleep(ENGINE_LOOP_SECS)


if __name__ == "__main__":
    asyncio.run(main())
