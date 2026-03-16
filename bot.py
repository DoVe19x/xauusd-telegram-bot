"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD SIGNAL BOT — Telegram + Render.com           ║
║  Stratégie : EMA200 + RSI14 crossover 50 + ATR SL/TP        ║
║  Timeframe  : 1H  |  Source : GC=F + fallback XAUUSD=X      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

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
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# On essaie plusieurs symboles dans l'ordre
SYMBOLS         = ["GC=F", "XAUUSD=X", "GLD"]
SYMBOL_DISPLAY  = "XAUUSD"
TIMEFRAME       = "1h"

EMA_PERIOD  = 200
RSI_PERIOD  = 14
ATR_PERIOD  = 14
ATR_MULT_SL = 2.0
RR_RATIO    = 2.0

CHECK_INTERVAL_SEC  = 300
DATA_PERIOD         = "30d"
MIN_CANDLES         = EMA_PERIOD + RSI_PERIOD + 10

# ─────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────
def validate_config() -> bool:
    if not TELEGRAM_TOKEN:
        log.error("❌ TELEGRAM_TOKEN manquant")
        return False
    if not TELEGRAM_CHAT_ID:
        log.error("❌ TELEGRAM_CHAT_ID manquant")
        return False
    log.info("✅ Configuration validée")
    return True

# ─────────────────────────────────────────────────────────────
#  DONNÉES — essaie plusieurs symboles
# ─────────────────────────────────────────────────────────────
def fetch_ohlcv() -> pd.DataFrame:
    """
    Essaie GC=F puis XAUUSD=X puis GLD.
    Retourne le premier DataFrame valide.
    """
    last_error = None

    for symbol in SYMBOLS:
        try:
            log.info(f"📡 Tentative téléchargement : {symbol}")
            df = yf.download(
                symbol,
                period=DATA_PERIOD,
                interval=TIMEFRAME,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            # Aplatir colonnes multi-index
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.dropna(inplace=True)

            if df.empty or len(df) < MIN_CANDLES:
                log.warning(f"⚠️  {symbol} : données insuffisantes ({len(df)} bougies)")
                continue

            log.info(f"✅ {symbol} : {len(df)} bougies — clôture : {float(df['Close'].iloc[-1]):.2f}")
            return df

        except Exception as e:
            last_error = e
            log.warning(f"⚠️  {symbol} échec : {e}")
            time.sleep(5)  # petite pause entre chaque tentative
            continue

    raise ConnectionError(f"Tous les symboles ont échoué. Dernière erreur : {last_error}")

# ─────────────────────────────────────────────────────────────
#  INDICATEURS
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
    df = df.copy()
    df["ema200"] = compute_ema(df["Close"], EMA_PERIOD)
    df["rsi14"]  = compute_rsi(df["Close"], RSI_PERIOD)
    df["atr14"]  = compute_atr(df, ATR_PERIOD)
    return df

# ─────────────────────────────────────────────────────────────
#  DÉTECTION DU SIGNAL
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

    log.info(f"📊 Prix: {price:.2f} | EMA200: {ema:.2f} | RSI: {rsi_now:.1f} (prev: {rsi_pre:.1f}) | ATR: {atr:.2f}")

    # LONG
    if price > ema and rsi_pre < 50 and rsi_now >= 50:
        return {
            "direction": "LONG",
            "price": round(price, 2),
            "sl": round(price - sl_dist, 2),
            "tp": round(price + tp_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp_dist": round(tp_dist, 2),
            "ema": round(ema, 2),
            "rsi": round(rsi_now, 1),
            "atr": round(atr, 2),
        }

    # SHORT
    if price < ema and rsi_pre > 50 and rsi_now <= 50:
        return {
            "direction": "SHORT",
            "price": round(price, 2),
            "sl": round(price + sl_dist, 2),
            "tp": round(price - tp_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp_dist": round(tp_dist, 2),
            "ema": round(ema, 2),
            "rsi": round(rsi_now, 1),
            "atr": round(atr, 2),
        }

    return None

# ─────────────────────────────────────────────────────────────
#  MESSAGES TELEGRAM
# ─────────────────────────────────────────────────────────────
def format_signal(sig: dict) -> str:
    is_long  = sig["direction"] == "LONG"
    emoji    = "🟢" if is_long else "🔴"
    arrow    = "📈" if is_long else "📉"
    now_utc  = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

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
        f"`{'─' * 32}`\n"
        f"_Stratégie : EMA200 \\+ RSI14 croisement 50 — 1H_\n"
        f"⚠️ _Gérez votre risque\\. Pas un conseil financier\\._"
    )

def format_startup() -> str:
    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    return (
        f"🤖 *Bot XAUUSD démarré\\!*\n\n"
        f"⏰ `{now_utc}`\n\n"
        f"📋 *Config :*\n"
        f"   • Paire : `{SYMBOL_DISPLAY}` \\| TF : `1H`\n"
        f"   • EMA : `{EMA_PERIOD}` \\| RSI : `{RSI_PERIOD}` \\(niveau 50\\)\n"
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
    log.info("  XAUUSD Signal Bot — démarrage")
    log.info("=" * 60)

    if not validate_config():
        log.error("⛔ Arrêt : config invalide")
        return

    send_telegram(format_startup())

    last_signal_direction: str | None = None
    error_count: int = 0

    log.info(f"🔄 Scan toutes les {CHECK_INTERVAL_SEC}s")

    while True:
        try:
            df     = fetch_ohlcv()
            df     = add_indicators(df)
            signal = detect_signal(df)
            error_count = 0  # reset si succès

            if signal is None:
                log.info("🔍 Pas de signal — conditions non réunies")

            elif signal["direction"] == last_signal_direction:
                log.info(f"⏭  Signal {signal['direction']} ignoré — doublon")

            else:
                log.info(f"🚨 SIGNAL : {signal['direction']} @ {signal['price']} | SL {signal['sl']} | TP {signal['tp']}")
                if send_telegram(format_signal(signal)):
                    last_signal_direction = signal["direction"]

        except ConnectionError as e:
            error_count += 1
            log.warning(f"⚠️  Erreur données ({error_count}) : {e}")
            # Pas de message Telegram pour les erreurs de données — pas de spam

        except Exception as e:
            error_count += 1
            log.error(f"💥 Erreur inattendue ({error_count}) : {e}")

        log.info(f"⏳ Prochain scan dans {CHECK_INTERVAL_SEC}s...")
        time.sleep(CHECK_INTERVAL_SEC)

if __name__ == "__main__":
    main()
