"""
╔══════════════════════════════════════════════════════════════╗
║         XAUUSD SIGNAL BOT — Telegram + Render.com           ║
║  Stratégie : EMA200 + RSI14 crossover 50 + ATR SL/TP        ║
║  Timeframe  : 1H  |  Paire : XAUUSD (GC=F Yahoo Finance)    ║
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
#  LOGGING — format structuré avec timestamp et niveau
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION — tout via variables d'environnement
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Symbole Yahoo Finance pour l'or (Gold Futures)
SYMBOL          = "GC=F"
SYMBOL_DISPLAY  = "XAUUSD"
TIMEFRAME       = "1h"

# Paramètres stratégie
EMA_PERIOD  = 200
RSI_PERIOD  = 14
ATR_PERIOD  = 14
ATR_MULT_SL = 2.0
RR_RATIO    = 2.0

# Boucle principale
CHECK_INTERVAL_SEC = 300   # toutes les 5 minutes
DATA_PERIOD        = "30d" # 30 jours de données pour avoir 200 bougies EMA


# ─────────────────────────────────────────────────────────────
#  VALIDATION AU DÉMARRAGE
# ─────────────────────────────────────────────────────────────
def validate_config() -> bool:
    """Vérifie que les variables d'environnement obligatoires sont présentes."""
    if not TELEGRAM_TOKEN:
        log.error("❌ Variable manquante : TELEGRAM_TOKEN non défini")
        return False
    if not TELEGRAM_CHAT_ID:
        log.error("❌ Variable manquante : TELEGRAM_CHAT_ID non défini")
        return False
    log.info("✅ Configuration validée")
    return True


# ─────────────────────────────────────────────────────────────
#  DONNÉES DE MARCHÉ
# ─────────────────────────────────────────────────────────────
def fetch_ohlcv() -> pd.DataFrame:
    """
    Télécharge les bougies horaires depuis Yahoo Finance.
    Retourne un DataFrame avec colonnes Open/High/Low/Close/Volume.
    Lève une exception si les données sont vides ou insuffisantes.
    """
    try:
        df = yf.download(
            SYMBOL,
            period=DATA_PERIOD,
            interval=TIMEFRAME,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise ConnectionError(f"Impossible de télécharger les données : {e}")

    if df is None or df.empty:
        raise ValueError("Données reçues vides — marché fermé ou symbole invalide.")

    # Aplatir les colonnes multi-index si yfinance les retourne ainsi
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    min_required = EMA_PERIOD + RSI_PERIOD + 10
    if len(df) < min_required:
        raise ValueError(
            f"Pas assez de données : {len(df)} bougies reçues, "
            f"minimum requis : {min_required}"
        )

    log.info(f"📥 {len(df)} bougies chargées — dernière clôture : {df['Close'].iloc[-1]:.2f}")
    return df


# ─────────────────────────────────────────────────────────────
#  CALCUL DES INDICATEURS  (pandas/numpy pur, sans TA-Lib)
# ─────────────────────────────────────────────────────────────
def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """EMA (Exponential Moving Average)."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    RSI (Relative Strength Index) méthode Wilder (EMA des gains/pertes).
    Retourne une série entre 0 et 100.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    # Éviter division par zéro
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # valeur neutre si indéfini


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    ATR (Average True Range) — mesure la volatilité.
    True Range = max(H-L, |H-Cprev|, |L-Cprev|)
    """
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
    """Calcule et ajoute toutes les colonnes d'indicateurs au DataFrame."""
    df = df.copy()
    df["ema200"] = compute_ema(df["Close"], EMA_PERIOD)
    df["rsi14"]  = compute_rsi(df["Close"], RSI_PERIOD)
    df["atr14"]  = compute_atr(df, ATR_PERIOD)
    return df


# ─────────────────────────────────────────────────────────────
#  LOGIQUE DE SIGNAL
# ─────────────────────────────────────────────────────────────
def detect_signal(df: pd.DataFrame) -> dict | None:
    """
    Analyse les deux dernières bougies et retourne un dict signal ou None.

    LONG  : Close > EMA200  ET  RSI[-2] < 50  ET  RSI[-1] >= 50  (crossover)
    SHORT : Close < EMA200  ET  RSI[-2] > 50  ET  RSI[-1] <= 50  (crossunder)

    Retourne un dict avec :
        direction, price, sl, tp, ema, rsi, atr, rr
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price   = float(last["Close"])
    ema     = float(last["ema200"])
    rsi_now = float(last["rsi14"])
    rsi_pre = float(prev["rsi14"])
    atr     = float(last["atr14"])

    sl_distance = ATR_MULT_SL * atr
    tp_distance = sl_distance * RR_RATIO

    # ── LONG ──────────────────────────────────────────────────
    if price > ema and rsi_pre < 50 and rsi_now >= 50:
        sl = round(price - sl_distance, 2)
        tp = round(price + tp_distance, 2)
        return {
            "direction" : "LONG",
            "price"     : round(price, 2),
            "sl"        : sl,
            "tp"        : tp,
            "sl_dist"   : round(sl_distance, 2),
            "tp_dist"   : round(tp_distance, 2),
            "ema"       : round(ema, 2),
            "rsi"       : round(rsi_now, 1),
            "atr"       : round(atr, 2),
            "rr"        : RR_RATIO,
        }

    # ── SHORT ─────────────────────────────────────────────────
    if price < ema and rsi_pre > 50 and rsi_now <= 50:
        sl = round(price + sl_distance, 2)
        tp = round(price - tp_distance, 2)
        return {
            "direction" : "SHORT",
            "price"     : round(price, 2),
            "sl"        : sl,
            "tp"        : tp,
            "sl_dist"   : round(sl_distance, 2),
            "tp_dist"   : round(tp_distance, 2),
            "ema"       : round(ema, 2),
            "rsi"       : round(rsi_now, 1),
            "atr"       : round(atr, 2),
            "rr"        : RR_RATIO,
        }

    return None


# ─────────────────────────────────────────────────────────────
#  FORMATAGE DU MESSAGE TELEGRAM
# ─────────────────────────────────────────────────────────────
def format_signal_message(sig: dict) -> str:
    """
    Construit le message Telegram en Markdown.
    Markdown Telegram : *gras*  _italique_  `code`
    """
    is_long   = sig["direction"] == "LONG"
    dir_emoji = "🟢" if is_long else "🔴"
    dir_arrow = "📈" if is_long else "📉"
    sl_emoji  = "🛑"
    tp_emoji  = "🎯"
    now_utc   = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")

    return (
        f"{dir_emoji} *SIGNAL {sig['direction']} — {SYMBOL_DISPLAY}* {dir_arrow}\n"
        f"`{'─' * 32}`\n"
        f"⏰ *Heure :* `{now_utc}`\n"
        f"⏱ *Timeframe :* `1H`\n\n"
        f"💰 *Entrée :*      `{sig['price']:,.2f} $`\n"
        f"{sl_emoji} *Stop Loss :*  `{sig['sl']:,.2f} $`  _\(−{sig['sl_dist']:,.2f} $\)_\n"
        f"{tp_emoji} *Take Profit :* `{sig['tp']:,.2f} $`  _\(\+{sig['tp_dist']:,.2f} $\)_\n\n"
        f"⚖️ *Ratio R/R :* `1:{sig['rr']}`  \|  *SL :* `{ATR_MULT_SL}× ATR`\n"
        f"`{'─' * 32}`\n"
        f"📊 *Indicateurs au signal :*\n"
        f"   • EMA200 : `{sig['ema']:,.2f}`\n"
        f"   • RSI14  : `{sig['rsi']}`\n"
        f"   • ATR14  : `{sig['atr']:,.2f}`\n"
        f"`{'─' * 32}`\n"
        f"_Stratégie : EMA200 \+ RSI14 croisement 50_\n"
        f"⚠️ _Gérez toujours votre risque\. Ce signal n'est pas un conseil financier\._"
    )


def format_startup_message() -> str:
    """Message de démarrage envoyé une fois au lancement du bot."""
    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    return (
        f"🤖 *Bot XAUUSD démarré avec succès\!*\n\n"
        f"⏰ `{now_utc}`\n\n"
        f"📋 *Configuration :*\n"
        f"   • Paire    : `{SYMBOL_DISPLAY}`\n"
        f"   • TF       : `1H`\n"
        f"   • EMA      : `{EMA_PERIOD}`\n"
        f"   • RSI      : `{RSI_PERIOD}` \(niveau 50\)\n"
        f"   • SL       : `{ATR_MULT_SL}× ATR{ATR_PERIOD}`\n"
        f"   • R/R      : `1:{RR_RATIO}`\n"
        f"   • Scan     : toutes les `5 min`\n\n"
        f"_En attente de signaux\.\.\._"
    )


def format_error_message(error: str) -> str:
    """Message d'erreur critique envoyé sur Telegram si le bot plante."""
    now_utc = datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M UTC")
    return (
        f"⚠️ *Erreur bot XAUUSD*\n"
        f"`{now_utc}`\n\n"
        f"```\n{error[:300]}\n```\n"
        f"_Le bot va réessayer dans 5 minutes\._"
    )


# ─────────────────────────────────────────────────────────────
#  ENVOI TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_telegram(text: str, parse_mode: str = "MarkdownV2") -> bool:
    """
    Envoie un message via l'API Telegram Bot.
    Retourne True si succès, False sinon.
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id"                  : TELEGRAM_CHAT_ID,
        "text"                     : text,
        "parse_mode"               : parse_mode,
        "disable_web_page_preview" : True,
    }

    try:
        response = requests.post(url, data=payload, timeout=15)
        response.raise_for_status()
        log.info("✅ Message Telegram envoyé avec succès")
        return True

    except requests.exceptions.HTTPError as e:
        log.error(f"❌ Erreur HTTP Telegram : {e} — réponse : {response.text[:200]}")
    except requests.exceptions.ConnectionError:
        log.error("❌ Erreur réseau — impossible de joindre Telegram")
    except requests.exceptions.Timeout:
        log.error("❌ Timeout — Telegram n'a pas répondu dans les 15s")
    except Exception as e:
        log.error(f"❌ Erreur inattendue Telegram : {e}")

    return False


# ─────────────────────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("  XAUUSD Signal Bot — démarrage")
    log.info("=" * 60)

    # Validation configuration
    if not validate_config():
        log.error("⛔ Arrêt : variables d'environnement manquantes.")
        return

    # Message de démarrage sur Telegram
    send_telegram(format_startup_message())

    # État interne — anti-doublon
    last_signal_direction: str | None = None
    consecutive_errors: int = 0
    MAX_CONSECUTIVE_ERRORS = 10  # alerte Telegram après 10 erreurs d'affilée

    log.info(f"🔄 Boucle démarrée — scan toutes les {CHECK_INTERVAL_SEC}s")

    while True:
        try:
            # ── 1. Récupération des données ──────────────────────
            df = fetch_ohlcv()

            # ── 2. Calcul des indicateurs ────────────────────────
            df = add_indicators(df)

            # ── 3. Détection du signal ───────────────────────────
            signal = detect_signal(df)

            # ── 4. Traitement du signal ──────────────────────────
            if signal is None:
                log.info("🔍 Aucun signal — conditions non réunies")

            elif signal["direction"] == last_signal_direction:
                log.info(
                    f"⏭  Signal {signal['direction']} ignoré — "
                    f"même direction que le précédent (anti-doublon)"
                )

            else:
                log.info(
                    f"🚨 SIGNAL DÉTECTÉ : {signal['direction']} "
                    f"@ {signal['price']} $ — "
                    f"SL {signal['sl']} $ / TP {signal['tp']} $"
                )
                msg = format_signal_message(signal)
                if send_telegram(msg):
                    last_signal_direction = signal["direction"]

            # Reset compteur d'erreurs après succès
            consecutive_errors = 0

        except (ConnectionError, ValueError) as e:
            consecutive_errors += 1
            log.warning(f"⚠️  Erreur récupérable ({consecutive_errors}) : {e}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log.error("🔥 Trop d'erreurs consécutives — notification Telegram")
                send_telegram(format_error_message(str(e)))
                consecutive_errors = 0  # reset pour ne pas spammer

        except Exception as e:
            consecutive_errors += 1
            log.error(f"💥 Erreur inattendue ({consecutive_errors}) : {e}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                send_telegram(format_error_message(str(e)))
                consecutive_errors = 0

        # ── 5. Attente avant prochain scan ───────────────────────
        log.info(f"⏳ Prochain scan dans {CHECK_INTERVAL_SEC}s...")
        time.sleep(CHECK_INTERVAL_SEC)


# ─────────────────────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
