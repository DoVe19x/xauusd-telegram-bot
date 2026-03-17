"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD SIGNAL BOT v4 — Twelve Data API              ║
║  Stratégie : EMA200 + RSI14 crossover 45/55 + ATR SL/TP     ║
║  Timeframe  : 1H  |  Paire : XAU/USD                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID", "")
TWELVEDATA_KEY    = os.environ.get("TWELVEDATA_KEY", "")

SYMBOL_DISPLAY    = "XAUUSD"
TWELVEDATA_SYMBOL = "XAU/USD"
TIMEFRAME         = "1h"
OUTPUT_SIZE       = 250

EMA_PERIOD        = 200
RSI_PERIOD        = 14
ATR_PERIOD        = 14
ATR_MULT_SL       = 2.0
RR_RATIO          = 2.0

# ── Niveaux RSI ──────────────────────────────────────────────
RSI_LONG_LEVEL    = 45   # LONG  : RSI croise AU-DESSUS de 45
RSI_SHORT_LEVEL   = 55   # SHORT : RSI croise EN-DESSOUS de 55

CHECK_INTERVAL_SEC = 300

# ─────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_config() -> bool:
    ok = True
    if not TELEGRAM_TOKEN:
        log.error("❌ TELEGRAM_TOKEN manquant"); ok = False
    if not TELEGRAM_CHAT_ID:
        log.error("❌ TELEGRAM_CHAT_ID manquant"); ok = False
    if not TWELVEDATA_KEY:
        log.error("❌ TWELVEDATA_KEY manquant"); ok = False
    if ok:
        log.info("✅ Configuration validée")
    return ok

# ─────────────────────────────────────────────────────────────
#  DONNÉES — Twelve Data
# ─────────────────────────────────────────────────────────────
def fetch_ohlcv() -> pd.DataFrame:
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol"     : TWELVEDATA_SYMBOL,
        "interval"   : TIMEFRAME,
        "outputsize" : OUTPUT_SIZE,
        "apikey"     : TWELVEDATA_KEY,
        "format"     : "JSON",
        "order"      : "ASC",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise ConnectionError(f"Erreur Twelve Data : {e}")

    if data.get("status") == "error":
        raise ValueError(f"Twelve Data : {data.get('message', 'erreur inconnue')}")

    values = data.get("values")
    if not values:
        raise ValueError("Twelve Data : aucune donnée reçue")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    for col in ["Open","High","Low","Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    min_req = EMA_PERIOD + RSI_PERIOD + 10
    if len(df) < min_req:
        raise ValueError(f"Données insuffisantes : {len(df)} bougies")

    log.info(f"✅ {len(df)} bougies — clôture : {float(df['Close'].iloc[-1]):.2f}")
    return df

# ─────────────────────────────────────────────────────────────
#  INDICATEURS  (pandas 3.0 compatible — pas de ChainedAssignment)
# ─────────────────────────────────────────────────────────────
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Utilise pd.assign pour éviter ChainedAssignmentWarning
    return df.assign(
        ema200 = compute_ema(df["Close"], EMA_PERIOD),
        rsi14  = compute_rsi(df["Close"], RSI_PERIOD),
        atr14  = compute_atr(df, ATR_PERIOD),
    )

# ─────────────────────────────────────────────────────────────
#  DÉTECTION DU SIGNAL — RSI 45/55
# ─────────────────────────────────────────────────────────────
def detect_signal(df: pd.DataFrame) -> dict | None:
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price   = float(last["Close"])
    ema     = float(last["ema200"])
    rsi_now = float(last["rsi14"])
    rsi_pre = float(prev["rsi14"])
    atr     = float(last["atr14"])

    sl_dist = ATR_MULT_SL * atr
    tp_dist = sl_dist * RR_RATIO

    log.info(
        f"📊 Prix: {price:.2f} | EMA200: {ema:.2f} | "
        f"RSI: {rsi_now:.1f} (prev: {rsi_pre:.1f}) | ATR: {atr:.2f} | "
        f"Trend: {'▲ BULL' if price > ema else '▼ BEAR'}"
    )

    # ── LONG : prix > EMA200 ET RSI croise AU-DESSUS de 45 ───
    if price > ema and rsi_pre < RSI_LONG_LEVEL and rsi_now >= RSI_LONG_LEVEL:
        return {
            "direction" : "LONG",
            "price"     : round(price, 2),
            "sl"        : round(price - sl_dist, 2),
            "tp"        : round(price + tp_dist, 2),
            "sl_dist"   : round(sl_dist, 2),
            "tp_dist"   : round(tp_dist, 2),
            "ema"       : round(ema, 2),
            "rsi"       : round(rsi_now, 1),
            "atr"       : round(atr, 2),
            "rsi_level" : RSI_LONG_LEVEL,
        }

    # ── SHORT : prix < EMA200 ET RSI croise EN-DESSOUS de 55 ─
    if price < ema and rsi_pre > RSI_SHORT_LEVEL and rsi_now <= RSI_SHORT_LEVEL:
        return {
            "direction" : "SHORT",
            "price"     : round(price, 2),
            "sl"        : round(price + sl_dist, 2),
            "tp"        : round(price - tp_dist, 2),
            "sl_dist"   : round(sl_dist, 2),
            "tp_dist"   : round(tp_dist, 2),
            "ema"       : round(ema, 2),
            "rsi"       : round(rsi_now, 1),
            "atr"       : round(atr, 2),
            "rsi_level" : RSI_SHORT_LEVEL,
        }

    return None

# ─────────────────────────────────────────────────────────────
#  MESSAGES TELEGRAM
# ─────────────────────────────────────────────────────────────
def format_signal(sig: dict) -> str:
    is_long = sig["direction"] == "LONG"
    emoji   = "🟢" if is_long else "🔴"
    arrow   = "📈" if is_long else "📉"
    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    return (
        f"{emoji} *SIGNAL {sig['direction']} — {SYMBOL_DISPLAY}* {arrow}\n"
        f"`{'─' * 32}`\n"
        f"⏰ *Heure :* `{now_utc}`\n"
        f"⏱ *Timeframe :* `1H`\n\n"
        f"💰 *Entrée :*       `{sig['price']:,.2f} $`\n"
        f"🛑 *Stop Loss :*   `{sig['sl']:,.2f} $`  _\\(\\-{sig['sl_dist']:,.2f} $\\)_\n"
        f"🎯 *Take Profit :* `{sig['tp']:,.2f} $`  _\\(\\+{sig['tp_dist']:,.2f} $\\)_\n\n"
        f"⚖️ *R/R :* `1:{RR_RATIO}`  \\|  *SL :* `{ATR_MULT_SL}× ATR`\n"
        f"`{'─' * 32}`\n"
        f"📊 *EMA200 :* `{sig['ema']:,.2f}`  \\|  *RSI14 :* `{sig['rsi']}`  \\|  *ATR :* `{sig['atr']:,.2f}`\n"
        f"🎯 *Niveau RSI déclenché :* `{sig['rsi_level']}`\n"
        f"`{'─' * 32}`\n"
        f"_Stratégie : EMA200 \\+ RSI14 croisement {sig['rsi_level']} — 1H_\n"
        f"⚠️ _Gérez votre risque\\. Pas un conseil financier\\._"
    )

def format_startup() -> str:
    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    return (
        f"🤖 *Bot XAUUSD v4 démarré\\!*\n\n"
        f"⏰ `{now_utc}`\n\n"
        f"📋 *Config :*\n"
        f"   • Paire : `{SYMBOL_DISPLAY}` \\| TF : `1H`\n"
        f"   • Source : `Twelve Data API`\n"
        f"   • EMA : `{EMA_PERIOD}`\n"
        f"   • RSI LONG : croise `{RSI_LONG_LEVEL}` \\| SHORT : croise `{RSI_SHORT_LEVEL}`\n"
        f"   • SL : `{ATR_MULT_SL}× ATR{ATR_PERIOD}` \\| R/R : `1:{RR_RATIO}`\n"
        f"   • Scan : toutes les `5 min`\n\n"
        f"_En attente de signaux\\.\\.\\._"
    )

def send_telegram(text: str) -> bool:
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id"                  : TELEGRAM_CHAT_ID,
        "text"                     : text,
        "parse_mode"               : "MarkdownV2",
        "disable_web_page_preview" : True,
    }
    try:
        r = requests.post(url, data=payload, timeout=15)
        r.raise_for_status()
        log.info("✅ Message Telegram envoyé")
        return True
    except Exception as e:
        log.error(f"❌ Telegram erreur : {e}")
        return False

# ─────────────────────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info(f"  XAUUSD Signal Bot v4 — RSI {RSI_LONG_LEVEL}/{RSI_SHORT_LEVEL}")
    log.info("=" * 60)

    if not validate_config():
        log.error("⛔ Arrêt : config invalide")
        return

    send_telegram(format_startup())

    last_signal_direction: str | None = None

    log.info(f"🔄 Scan toutes les {CHECK_INTERVAL_SEC}s")

    while True:
        try:
            df     = fetch_ohlcv()
            df     = add_indicators(df)
            signal = detect_signal(df)

            if signal is None:
                log.info("🔍 Pas de signal — conditions non réunies")
            elif signal["direction"] == last_signal_direction:
                log.info(f"⏭  Doublon {signal['direction']} ignoré")
            else:
                log.info(
                    f"🚨 SIGNAL {signal['direction']} @ {signal['price']} "
                    f"| SL {signal['sl']} | TP {signal['tp']}"
                )
                if send_telegram(format_signal(signal)):
                    last_signal_direction = signal["direction"]

        except Exception as e:
            log.warning(f"⚠️  Erreur : {e}")

        log.info(f"⏳ Prochain scan dans {CHECK_INTERVAL_SEC}s...")
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
