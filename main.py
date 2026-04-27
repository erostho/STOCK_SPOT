import os
import sys
import json
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import ta
from vnstock import Vnstock, Quote, register_user

# ============================================================
# WEEKLY STOCK WATCHLIST BOT
# - Nguồn universe: Google Sheet CSV (SHEET_CSV_URL)
# - Nguồn dữ liệu free: vnstock (price/FA/VNINDEX)
# - API key Community: VNSTOCK_API_KEY (60 req/phút)
# - Telegram env: TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
# - Output: 3 danh mục TOP 5
# ============================================================

# ---------------- ENV ----------------
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
VNSTOCK_API_KEY = os.getenv("VNSTOCK_API_KEY", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------- cache files ----------
PRICE_CACHE_FILE = os.path.join(CACHE_DIR, "price_cache.json")
FA_CACHE_FILE = os.path.join(CACHE_DIR, "fa_cache_v2.json")
UNIVERSE_CACHE_FILE = os.path.join(CACHE_DIR, "universe_cache.json")

# ---------- throttling ----------
RATE_LIMIT_PER_MIN = int(os.getenv("VNSTOCK_RATE_LIMIT_PER_MIN", "45"))  # giữ đệm dưới 60
REQUEST_TIMESTAMPS = deque()

# ---------- user knobs ----------
TOP_N_PER_BUCKET = 5
PRICE_HISTORY_BARS = 320
PRICE_TTL_SEC = 24 * 3600
FA_TTL_SEC = 7 * 24 * 3600
UNIVERSE_TTL_SEC = 24 * 3600

PENNY_MAX_PRICE = 15_000
SHORT_MAX_PRICE = 100_000
LONG_MAX_PRICE = 100_000

PENNY_LIQ_MIN = 3e9
SHORT_LIQ_MIN = 7e9
LONG_LIQ_MIN = 3e9

# ---------- batch scan ----------
BATCH_SIZE = int(os.getenv("STOCK_SCAN_BATCH_SIZE", "510"))
BATCH_DELAY_SEC = int(os.getenv("STOCK_SCAN_BATCH_DELAY_SEC", "90"))

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# AUTH + CACHE + RATE LIMIT
# ============================================================

def configure_vnstock_auth():
    """Đăng ký API key nếu có. Không có thì vẫn chạy guest mode."""
    if not VNSTOCK_API_KEY:
        log("⚠️ VNSTOCK_API_KEY chưa có -> chạy guest mode (dễ chạm limit hơn).")
        return
    try:
        register_user(api_key=VNSTOCK_API_KEY)
        log("✅ Đã đăng ký VNSTOCK_API_KEY (Community/free nếu key hợp lệ).")
    except Exception as e:
        log(f"⚠️ Không đăng ký được API key vnstock: {e}")


def rate_limit_wait():
    now = time.time()
    while REQUEST_TIMESTAMPS and now - REQUEST_TIMESTAMPS[0] > 60:
        REQUEST_TIMESTAMPS.popleft()

    if len(REQUEST_TIMESTAMPS) >= RATE_LIMIT_PER_MIN:
        sleep_for = max(1.0, 60 - (now - REQUEST_TIMESTAMPS[0]) + 0.2)
        log(f"⏳ Chạm ngưỡng nội bộ {RATE_LIMIT_PER_MIN}/phút -> ngủ {sleep_for:.1f}s")
        time.sleep(sleep_for)

    REQUEST_TIMESTAMPS.append(time.time())


def cache_load(path: str, ttl_sec: int):
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path) < ttl_sec):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _json_safe(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

def cache_save(path: str, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(obj), f, ensure_ascii=False)
    except Exception as e:
        log(f"⚠️ cache_save lỗi {os.path.basename(path)}: {e}")


# ============================================================
# SHEET UNIVERSE
# ============================================================

def get_tickers_from_sheet() -> List[str]:
    if not SHEET_CSV_URL:
        log("❌ SHEET_CSV_URL chưa cấu hình.")
        return []

    cached = cache_load(UNIVERSE_CACHE_FILE, UNIVERSE_TTL_SEC)
    if cached and cached.get("tickers"):
        return cached["tickers"]

    try:
        df = pd.read_csv(SHEET_CSV_URL)
    except Exception as e:
        log(f"❌ Lỗi đọc Google Sheet CSV: {e}")
        return []

    col_ticker = None
    for c in df.columns:
        key = str(c).strip().lower()
        if key in ["mã", "ma", "ticker", "symbol", "code"]:
            col_ticker = c
            break
    if col_ticker is None:
        col_ticker = df.columns[0]

    tickers = (
        df[col_ticker]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("NAN", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    tickers = sorted(set([tk for tk in tickers if tk]))
    cache_save(UNIVERSE_CACHE_FILE, {"tickers": tickers})
    log(f"✅ Universe từ sheet: {len(tickers)} mã")
    return tickers


# ============================================================
# VNSTOCK DATA ACCESS
# ============================================================

def _safe_float(v, default=None):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _find_col(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    kws = [k.lower().replace(" ", "").replace("_", "") for k in keywords]
    for col in df.columns:
        key = str(col).lower().replace(" ", "").replace("_", "")
        if any(k in key for k in kws):
            return col
    return None


def _vns_call(fn, *args, retries: int = 2, **kwargs):
    last_error = None
    for attempt in range(retries + 1):
        try:
            rate_limit_wait()
            return fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < retries:
                sleep_s = 1.2 * (attempt + 1)
                log(f"⚠️ vnstock call lỗi, retry {attempt+1}/{retries}: {e}")
                time.sleep(sleep_s)
    raise last_error


def get_price_history(ticker: str, length: int = PRICE_HISTORY_BARS) -> pd.DataFrame:
    cache_all = cache_load(PRICE_CACHE_FILE, PRICE_TTL_SEC) or {}
    if ticker in cache_all:
        try:
            df = pd.DataFrame(cache_all[ticker])
            if not df.empty:
                df["time"] = pd.to_datetime(df["time"])
                return df
        except Exception:
            pass

    # ưu tiên KBS cho quote history theo docs vnstock
    quote = Quote(symbol=ticker, source="KBS")
    df = _vns_call(quote.history, length=str(length), interval="1D")
    if df is None or df.empty:
        # fallback VCI
        quote = Quote(symbol=ticker, source="VCI")
        df = _vns_call(quote.history, length=str(length), interval="1D")

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if "time" not in df.columns:
        for c in ["date", "tradingDate"]:
            if c in df.columns:
                df["time"] = pd.to_datetime(df[c])
                break
    else:
        df["time"] = pd.to_datetime(df["time"])

    rename_map = {}
    for c in df.columns:
        low = str(c).lower()
        if low == "open":
            rename_map[c] = "open"
        elif low == "high":
            rename_map[c] = "high"
        elif low == "low":
            rename_map[c] = "low"
        elif low == "close":
            rename_map[c] = "close"
        elif low == "volume":
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)

    need_cols = ["time", "open", "high", "low", "close", "volume"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[need_cols].dropna(subset=["close"]).copy()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)

    df_cache = df.copy()
    df_cache["time"] = df_cache["time"].astype(str)
    cache_all[ticker] = df_cache.to_dict(orient="records")
    cache_save(PRICE_CACHE_FILE, cache_all)
    return df


def get_fa_data(ticker: str) -> Dict:
    cache_all = cache_load(FA_CACHE_FILE, FA_TTL_SEC) or {}
    if ticker in cache_all:
        return cache_all[ticker]

    def empty_fa(source_used=None, error=None) -> Dict:
        return {
            "ticker": ticker,
            "eps": None,
            "roe": None,
            "pe": None,
            "pb": None,
            "de": None,
            "rev_growth": None,
            "np_growth": None,
            "latest_net_profit": None,
            "fa_quality_score": 0,
            "fa_ok": False,
            "fa_source": source_used,
            "fa_error": error,
        }

    def read_fa_from_source(source: str) -> Dict:
        result = empty_fa(source_used=source, error=None)

        stock = Vnstock().stock(symbol=ticker, source=source)
        finance = stock.finance

        # ---------- RATIO ----------
        ratio_df = _vns_call(finance.ratio, period="year", lang="vi", dropna=True)

        if ratio_df is not None and not ratio_df.empty:
            row = ratio_df.iloc[-1]

            eps_col = _find_col(ratio_df, ["eps"])
            roe_col = _find_col(ratio_df, ["roe"])
            pe_col = _find_col(ratio_df, ["p/e", "pe"])
            pb_col = _find_col(ratio_df, ["p/b", "pb"])
            de_col = _find_col(ratio_df, ["nợ/vốn", "debttoequity", "debt/equity", "d/e"])

            if eps_col:
                result["eps"] = _safe_float(row.get(eps_col))
            if roe_col:
                result["roe"] = _safe_float(row.get(roe_col))
            if pe_col:
                result["pe"] = _safe_float(row.get(pe_col))
            if pb_col:
                result["pb"] = _safe_float(row.get(pb_col))
            if de_col:
                result["de"] = _safe_float(row.get(de_col))

        # ---------- INCOME ----------
        income_df = _vns_call(finance.income_statement, period="year", dropna=True)

        if income_df is not None and not income_df.empty:
            income_df = income_df.tail(4).copy()
            np_col = _find_col(income_df, ["lnst", "loinhuansauthue", "netprofit"])
            rev_col = _find_col(income_df, ["doanhthu", "revenue", "sales"])

            if np_col:
                np_list = pd.to_numeric(income_df[np_col], errors="coerce").dropna().tolist()
                if np_list:
                    result["latest_net_profit"] = _safe_float(np_list[-1])
                if len(np_list) >= 2 and np_list[-2] not in [0, None]:
                    result["np_growth"] = ((np_list[-1] - np_list[-2]) / abs(np_list[-2])) * 100

            if rev_col:
                rev_list = pd.to_numeric(income_df[rev_col], errors="coerce").dropna().tolist()
                if len(rev_list) >= 2 and rev_list[-2] not in [0, None]:
                    result["rev_growth"] = ((rev_list[-1] - rev_list[-2]) / abs(rev_list[-2])) * 100

        # ---------- FA QUALITY ----------
        score = 0
        if result["eps"] is not None and result["eps"] > 0:
            score += 1
        if result["roe"] is not None and result["roe"] >= 10:
            score += 1
        if result["pe"] is not None and 0 < result["pe"] < 18:
            score += 1
        if result["de"] is not None and result["de"] < 1.5:
            score += 1
        if result["np_growth"] is not None and result["np_growth"] > 0:
            score += 1

        result["fa_quality_score"] = score

        result["fa_ok"] = any([
            result["eps"] is not None,
            result["roe"] is not None,
            result["pe"] is not None,
            result["pb"] is not None,
            result["latest_net_profit"] is not None,
        ])

        return result

    # ========================================================
    # FALLBACK FLOW: VCI -> TCBS -> FA None
    # ========================================================
    errors = []

    for source in ["VCI", "KBS", "FMP"]:
        try:
            fa = read_fa_from_source(source)

            if fa.get("fa_ok"):
                log(f"✅ FA {ticker}: lấy được từ {source}")
                cache_all[ticker] = fa
                cache_save(FA_CACHE_FILE, cache_all)
                return fa

            errors.append(f"{source}: empty FA")

        except Exception as e:
            err = str(e)
            errors.append(f"{source}: {err}")

            if err == "data" or "'data'" in err:
                log(f"⚠️ FA {ticker}: {source} lỗi thiếu key 'data' -> thử nguồn khác")
            else:
                log(f"⚠️ FA {ticker}: {source} lỗi {err} -> thử nguồn khác")

            continue

    # Nếu cả VCI/KBS/FMP đều fail
    final_error = " | ".join(errors)
    log(f"⚠️ FA {ticker}: cả VCI/KBS/FMP đều lỗi hoặc rỗng -> dùng FA rỗng | {final_error}")

    return empty_fa(source_used=None, error=final_error)


def get_market_regime() -> Tuple[int, str, str]:
    try:
        df = get_price_history("VNINDEX", length=260)
        if df is None or df.empty or len(df) < 120:
            return 0, "TRUNG TÍNH", "Thị trường chưa đủ rõ xu hướng, ưu tiên chọn mã có nền giá đẹp."

        close = df["close"].astype(float)
        ma50 = close.rolling(50).mean().iloc[-1]
        ma100 = close.rolling(100).mean().iloc[-1]
        last = float(close.iloc[-1])

        if pd.notna(ma50) and pd.notna(ma100):
            if last > ma50 > ma100:
                return 1, "TÍCH CỰC", "Dòng tiền thị trường ở trạng thái thuận lợi hơn, có thể ưu tiên mã khỏe và nền tích lũy đẹp."
            if last < ma50 < ma100:
                return -1, "THẬN TRỌNG", "Thị trường chung yếu hơn, nên ưu tiên chọn lọc kỹ và tránh mua đuổi."
        return 0, "TRUNG TÍNH", "Thị trường chưa quá rõ xu hướng, nên ưu tiên cổ phiếu có điểm mua gần hỗ trợ."
    except Exception as e:
        log(f"⚠️ market regime lỗi: {e}")
        return 0, "TRUNG TÍNH", "Không đọc được bối cảnh VNINDEX, ưu tiên quản trị vị thế vừa phải."


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.empty or len(df) < 30:
        return df

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"].fillna(0)

    df["ma20"] = close.rolling(20).mean()
    df["ma50"] = close.rolling(50).mean()
    df["ma100"] = close.rolling(100).mean()
    df["ma200"] = close.rolling(200).mean()
    df["vol_ma20"] = vol.rolling(20).mean()
    df["value"] = close * 1000 * vol
    df["value_ma20"] = df["value"].rolling(20).mean()

    try:
        df["rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx"] = adx.adx()
        df["di_pos"] = adx.adx_pos()
        df["di_neg"] = adx.adx_neg()
    except Exception:
        df["rsi"] = pd.NA
        df["adx"] = pd.NA
        df["di_pos"] = pd.NA
        df["di_neg"] = pd.NA

    df["hh20"] = close.rolling(20).max()
    df["ll20"] = close.rolling(20).min()
    df["hh60"] = close.rolling(60).max()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    return df



def summarize_features(ticker: str, df: pd.DataFrame, fa: Dict, regime: int) -> Optional[Dict]:
    if df is None or df.empty or len(df) < 120:
        return None

    df = add_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = _safe_float(last["close"])
    ma20 = _safe_float(last["ma20"])
    ma50 = _safe_float(last["ma50"])
    ma100 = _safe_float(last["ma100"])
    ma200 = _safe_float(last["ma200"])
    close_vnd = close * 1000 if close is not None else None
    ma20_vnd = ma20 * 1000 if ma20 is not None else None
    ma50_vnd = ma50 * 1000 if ma50 is not None else None
    ma100_vnd = ma100 * 1000 if ma100 is not None else None
    ma200_vnd = ma200 * 1000 if ma200 is not None else None
    rsi = _safe_float(last["rsi"])
    adx = _safe_float(last["adx"])
    di_pos = _safe_float(last["di_pos"])
    di_neg = _safe_float(last["di_neg"])
    value20 = _safe_float(last["value_ma20"], 0) or 0
    vol_ratio = _safe_float(last["vol_ratio"], 0) or 0
    hh20_prev = _safe_float(df["close"].iloc[-21:-1].max())
    hh60_prev = _safe_float(df["close"].iloc[-61:-1].max())

    if not close_vnd or not ma20_vnd:
        return None
    hh20_prev_vnd = hh20_prev * 1000 if hh20_prev is not None else None
    hh60_prev_vnd = hh60_prev * 1000 if hh60_prev is not None else None
    
    dist_ma20 = abs(close_vnd - ma20_vnd) / ma20_vnd * 100 if ma20_vnd else None
    dist_ma50 = abs(close_vnd - ma50_vnd) / ma50_vnd * 100 if ma50_vnd else None
    trend_up_short = bool(ma20_vnd and ma50_vnd and close_vnd > ma20_vnd and ma20_vnd >= ma50_vnd)
    trend_up_mid = bool(ma50_vnd and ma100_vnd and close_vnd > ma50_vnd and ma50_vnd >= ma100_vnd)
    stage2 = bool(ma20_vnd and ma50_vnd and ma100_vnd and close_vnd > ma20_vnd > ma50_vnd > ma100_vnd)
    breakout20 = bool(hh20_prev_vnd and close_vnd > hh20_prev_vnd)
    breakout60 = bool(hh60_prev_vnd and close_vnd > hh60_prev_vnd)
    pullback_uptrend = bool(trend_up_short and dist_ma20 is not None and dist_ma20 <= 4)
    retest_breakout = bool(prev["close"] > df["close"].iloc[-22:-2].max() and dist_ma20 is not None and dist_ma20 <= 5)
    rsi_ok = bool(rsi and rsi > 50)
    adx_ok = bool(adx and adx > 18 and (di_pos or 0) > (di_neg or 0))
    near_ma20 = 2 if dist_ma20 is not None and dist_ma20 <= 3 else (1 if dist_ma20 is not None and dist_ma20 <= 6 else 0)
    quality_entry = 1 if dist_ma20 is not None and dist_ma20 <= 8 else 0

    return {
        "ticker": ticker,
        "price": round(close_vnd, 0),
        "close": close_vnd,
        "ma20": ma20_vnd,
        "ma50": ma50_vnd,
        "ma100": ma100_vnd,
        "ma200": ma200_vnd,
        "rsi": rsi,
        "adx": adx,
        "di_pos": di_pos,
        "di_neg": di_neg,
        "value20": value20,
        "vol_ratio": vol_ratio,
        "dist_ma20": dist_ma20,
        "dist_ma50": dist_ma50,
        "trend_up_short": trend_up_short,
        "trend_up_mid": trend_up_mid,
        "stage2": stage2,
        "breakout20": breakout20,
        "breakout60": breakout60,
        "pullback_uptrend": pullback_uptrend,
        "retest_breakout": retest_breakout,
        "rsi_ok": rsi_ok,
        "adx_ok": adx_ok,
        "near_ma20_bonus": near_ma20,
        "entry_quality_bonus": quality_entry,
        "regime": regime,
        **fa,
    }


# ============================================================
# SCREENERS
# ============================================================

def score_penny(x: Dict) -> Optional[Dict]:
    price = x["price"]
    if not (price and price < PENNY_MAX_PRICE):
        return None

    value20 = x.get("value20") or 0
    roe = x.get("roe")
    eps = x.get("eps")
    pe = x.get("pe")
    de = x.get("de")
    latest_np = x.get("latest_net_profit")
    np_growth = x.get("np_growth")
    rev_growth = x.get("rev_growth")
    dist_ma20 = x.get("dist_ma20")
    close = x.get("close")
    ma20 = x.get("ma20")
    ma50 = x.get("ma50")

    # -------------------------
    # EARLY REJECT: tránh penny quá rác / quá chết
    # -------------------------
    if value20 < 1.5e9:
        return None

    if (eps is not None and eps <= 0) and (latest_np is not None and latest_np <= 0):
        return None

    if roe is not None and roe < 2:
        return None

    if de is not None and de >= 4:
        return None

    if np_growth is not None and np_growth < -30:
        return None

    if close is not None:
        if ma20 is not None and ma50 is not None:
            if close < ma20 * 0.85 and close < ma50 * 0.85:
                return None
        elif ma20 is not None and close < ma20 * 0.82:
            return None

    # -------------------------
    # NỀN TỐI THIỂU: phải có ít nhất 2 tín hiệu "sống"
    # -------------------------
    base_signals = 0
    if eps is not None and eps > 0:
        base_signals += 1
    if latest_np is not None and latest_np > 0:
        base_signals += 1
    if roe is not None and roe >= 3:
        base_signals += 1
    if (np_growth is not None and np_growth > 0) or (rev_growth is not None and rev_growth > 0):
        base_signals += 1

    if base_signals < 2:
        return None

    # -------------------------
    # SCORING
    # -------------------------
    score = 0.0
    reasons = []

    # A. Nền cơ bản tối thiểu
    if eps is not None and eps > 0:
        score += 1.0
        reasons.append(f"EPS dương ({eps:.0f}), doanh nghiệp chưa rơi vào trạng thái quá yếu")

    if latest_np is not None and latest_np > 0:
        score += 0.8
        if len(reasons) < 3:
            reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")

    if roe is not None:
        if roe >= 8:
            score += 1.0
            if len(reasons) < 3:
                reasons.append(f"ROE {roe:.1f}% ở mức khá trong nhóm cổ phiếu giá thấp")
        elif roe >= 5:
            score += 0.7
        elif roe >= 3:
            score += 0.4

    # B. Dấu hiệu cải thiện
    if np_growth is not None:
        if np_growth > 15:
            score += 1.2
            if len(reasons) < 3:
                reasons.append("Lợi nhuận đang cải thiện khá rõ")
        elif np_growth > 0:
            score += 0.7
            if len(reasons) < 3:
                reasons.append("Lợi nhuận có dấu hiệu hồi phục")

    if rev_growth is not None:
        if rev_growth > 10:
            score += 0.8
        elif rev_growth > 0:
            score += 0.4

    # C. Thanh khoản
    if value20 >= 5e9:
        score += 1.5
        if len(reasons) < 3:
            reasons.append("Thanh khoản khá tốt so với mặt bằng nhóm cổ phiếu giá thấp")
    elif value20 >= 3e9:
        score += 1.0
    else:
        score += 0.5

    # D. Vị trí giá
    if dist_ma20 is not None:
        if dist_ma20 <= 5:
            score += 1.0
            if len(reasons) < 3:
                reasons.append("Giá đang ở vùng nền tương đối gần hỗ trợ")
        elif dist_ma20 <= 10:
            score += 0.5

    if close is not None and ma20 is not None and close >= ma20:
        score += 0.5

    if close is not None and ma50 is not None and close >= ma50:
        score += 0.5

    # E. Bonus định giá
    if pe is not None:
        if 0 < pe < 12:
            score += 0.5
        elif 12 <= pe < 18:
            score += 0.2

    # -------------------------
    # PENALTY
    # -------------------------
    if np_growth is not None and np_growth < 0:
        score -= 0.5

    if close is not None and ma20 is not None and close < ma20:
        score -= 0.3

    if close is not None and ma50 is not None and close < ma50:
        score -= 0.5

    if de is not None and de >= 3:
        score -= 0.5

    if pe is not None and pe > 20:
        score -= 0.5

    if x.get("regime", 0) > 0:
        score += 0.4

    score = max(0.0, min(10.0, round(score, 1)))

    # -------------------------
    # LABEL + RISK
    # -------------------------
    if np_growth is not None and np_growth > 15:
        label = "Hồi phục đáng chú ý"
        risk = "Penny biến động mạnh, chỉ phù hợp giải ngân nhỏ và chia lệnh."
    elif score >= 6.0:
        label = "Tích lũy đầu cơ có chọn lọc"
        risk = "Có thể theo dõi tích lũy từng phần, nhưng không phù hợp dồn tỷ trọng lớn."
    else:
        label = "Theo dõi thêm"
        risk = "Mã có vài tín hiệu tích cực nhưng chưa đủ mạnh, nên ưu tiên quan sát trước."

    buy_low = round(max(0, price * 0.95), 2)
    buy_high = round(price * 1.03, 2)

    if len(reasons) < 3 and roe is not None:
        reasons.append(f"ROE {roe:.1f}% vẫn ở mức chấp nhận được trong nhóm penny")
    if len(reasons) < 3 and value20 >= 1.5e9:
        reasons.append("Thanh khoản vẫn đủ để tiếp tục theo dõi")
    if len(reasons) < 3:
        reasons.append("Phù hợp kiểu thăm dò có chọn lọc hơn là giải ngân mạnh")

    return {
        **x,
        "score": score,
        "label": label,
        "buy_zone": f"{buy_low} - {buy_high}",
        "reasons": reasons[:3],
        "risk_note": risk,
    }


def score_short_term(x: Dict) -> Optional[Dict]:
    price = x["price"]
    if not (price and price < SHORT_MAX_PRICE):
        return None

    value20 = x.get("value20") or 0
    roe = x.get("roe")
    eps = x.get("eps")
    pe = x.get("pe")
    pb = x.get("pb")
    de = x.get("de")
    latest_np = x.get("latest_net_profit")
    np_growth = x.get("np_growth")
    rev_growth = x.get("rev_growth")
    dist_ma20 = x.get("dist_ma20")
    dist_ma50 = x.get("dist_ma50")
    close = x.get("close")
    ma20 = x.get("ma20")
    ma50 = x.get("ma50")
    rsi = x.get("rsi")
    adx = x.get("adx")
    di_pos = x.get("di_pos")
    di_neg = x.get("di_neg")

    # -------------------------
    # EARLY REJECT
    # -------------------------
    if value20 < 5e9:
        return None

    if eps is not None and eps <= 0:
        return None

    if latest_np is not None and latest_np <= 0:
        return None

    if roe is not None and roe < 7:
        return None

    if de is not None and de >= 3:
        return None

    if np_growth is not None and rev_growth is not None:
        if np_growth < -20 and rev_growth < 0:
            return None

    if pe is not None and pe > 25 and (np_growth is None or np_growth <= 0):
        return None

    if close is not None and ma50 is not None and close < ma50 * 0.93:
        return None

    # -------------------------
    # SCORING
    # -------------------------
    score = 0.0
    reasons = []

    # A. Chất lượng cơ bản vừa đủ
    if eps is not None and eps > 0:
        score += 0.8
        reasons.append(f"EPS dương ({eps:.0f}), nền doanh nghiệp không quá yếu")

    if latest_np is not None and latest_np > 0:
        score += 0.7
        if len(reasons) < 3:
            reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")

    if roe is not None:
        if roe >= 15:
            score += 1.0
            if len(reasons) < 3:
                reasons.append(f"ROE {roe:.1f}% ở mức tốt")
        elif roe >= 10:
            score += 0.7
        elif roe >= 7:
            score += 0.4

    # B. Tăng trưởng / hồi phục
    growth_flag = False
    if np_growth is not None:
        if np_growth > 20:
            score += 1.2
            growth_flag = True
            if len(reasons) < 3:
                reasons.append("Lợi nhuận đang tăng trưởng tích cực")
        elif np_growth > 0:
            score += 0.7
            growth_flag = True
            if len(reasons) < 3:
                reasons.append("Lợi nhuận có dấu hiệu cải thiện")

    if rev_growth is not None:
        if rev_growth > 15:
            score += 0.8
        elif rev_growth > 0:
            score += 0.4

    # C. Định giá
    if pe is not None:
        if 0 < pe <= 15:
            score += 1.0
        elif 15 < pe <= 20:
            score += 0.6

    if pb is not None:
        if 0 < pb <= 2:
            score += 0.5
        elif pb <= 3:
            score += 0.2

    # D. Thanh khoản
    if value20 >= 15e9:
        score += 1.5
        if len(reasons) < 3:
            reasons.append("Thanh khoản mạnh, thuận lợi cho danh mục 3-6 tháng")
    elif value20 >= 10e9:
        score += 1.0
    else:
        score += 0.5

    # E. Xu hướng giá trung hạn
    if close is not None and ma20 is not None and ma50 is not None:
        if close > ma20 > ma50:
            score += 1.0
        elif close > ma50:
            score += 0.7

    if dist_ma20 is not None and dist_ma20 <= 8:
        score += 0.5

    if dist_ma50 is not None and dist_ma50 <= 10:
        score += 0.5

    # F. Xác nhận sức mạnh giá
    if rsi is not None and rsi >= 55:
        score += 0.3

    if adx is not None and adx >= 20 and (di_pos is not None and di_neg is not None) and di_pos > di_neg:
        score += 0.5

    # -------------------------
    # PENALTY
    # -------------------------
    if pe is not None and pe > 22:
        score -= 0.5

    if dist_ma20 is not None and dist_ma20 > 12:
        score -= 0.5

    if close is not None and ma20 is not None and close < ma20:
        score -= 0.3

    if close is not None and ma50 is not None and close < ma50:
        score -= 0.7

    if np_growth is not None and np_growth <= 0:
        score -= 0.5

    if rev_growth is not None and rev_growth <= 0:
        score -= 0.3

    if de is not None:
        if de < 1:
            score += 0.4
        elif de < 2:
            score += 0.2

    if x.get("regime", 0) > 0:
        score += 0.5

    score = max(0.0, min(10.0, round(score, 1)))

    # -------------------------
    # LABEL + PLAN + RISK
    # -------------------------
    if growth_flag and score >= 7.0:
        setup_type = "Tăng trưởng 3-6 tháng"
        entry_plan = "Ưu tiên tích lũy từng phần khi giá còn giữ trên vùng hỗ trợ trung hạn."
    elif score >= 6.0:
        setup_type = "Hồi phục trung hạn"
        entry_plan = "Có thể mua từng phần khi giá giữ được MA20/MA50 và không tăng nóng quá nhanh."
    else:
        setup_type = "Tích lũy chờ tăng"
        entry_plan = "Ưu tiên theo dõi thêm, chỉ giải ngân khi giá vẫn giữ được xu hướng thuận."

    risk = "Danh mục 3-6 tháng vẫn cần theo dõi thêm kết quả kinh doanh và phản ứng giá quanh hỗ trợ trung hạn."

    buy_anchor = ma20 or ma50 or price
    buy_low = round(buy_anchor * 0.97, 2)
    buy_high = round(buy_anchor * 1.02, 2)

    if len(reasons) < 3 and pe is not None and pe > 0:
        reasons.append(f"PE {pe:.1f}, định giá chưa quá căng")
    if len(reasons) < 3 and close is not None and ma50 is not None and close > ma50:
        reasons.append("Giá vẫn đang duy trì trên nền hỗ trợ trung hạn")
    if len(reasons) < 3:
        reasons.append("Phù hợp cho góc nhìn 1-2 quý tới hơn là lướt sóng ngắn")

    return {
        **x,
        "score": score,
        "setup_type": setup_type,
        "entry_plan": entry_plan,
        "buy_zone": f"{buy_low} - {buy_high}",
        "reasons": reasons[:3],
        "risk_note": risk,
    }

def score_long_term(x: Dict) -> Optional[Dict]:
    price = x["price"]
    if not (price and price < LONG_MAX_PRICE):
        return None

    value20 = x.get("value20") or 0
    roe = x.get("roe")
    eps = x.get("eps")
    pe = x.get("pe")
    pb = x.get("pb")
    de = x.get("de")
    latest_np = x.get("latest_net_profit")
    np_growth = x.get("np_growth")
    rev_growth = x.get("rev_growth")
    dist_ma50 = x.get("dist_ma50")
    close = x.get("close")
    ma50 = x.get("ma50")
    ma100 = x.get("ma100")
    ma200 = x.get("ma200")

    # -------------------------
    # EARLY REJECT
    # -------------------------
    if value20 < 2e9:
        return None

    if eps is None or eps <= 0:
        return None

    if latest_np is None or latest_np <= 0:
        return None

    if roe is None or roe < 8:
        return None

    if de is not None and de >= 2.5:
        return None

    if pe is None or pe <= 0 or pe > 22:
        return None

    if np_growth is not None and np_growth < -15:
        return None

    if close is not None and ma100 is not None and close < ma100 * 0.90:
        return None

    if close is not None and ma200 is not None and close < ma200 * 0.90:
        return None

    # -------------------------
    # SCORING
    # -------------------------
    score = 0.0
    reasons = []

    # A. Chất lượng doanh nghiệp
    score += 0.8  # EPS dương
    reasons.append(f"EPS dương ({eps:.0f}), doanh nghiệp vẫn tạo ra lợi nhuận")

    score += 0.7  # lợi nhuận gần nhất dương
    if len(reasons) < 3:
        reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")

    if roe is not None:
        if roe >= 18:
            score += 1.5
            if len(reasons) < 3:
                reasons.append(f"ROE {roe:.1f}% ở mức rất tốt")
        elif roe >= 14:
            score += 1.1
            if len(reasons) < 3:
                reasons.append(f"ROE {roe:.1f}% ở mức tốt")
        elif roe >= 10:
            score += 0.7
        elif roe >= 8:
            score += 0.3

    # B. Tăng trưởng bền
    growth_good = False
    if np_growth is not None:
        if np_growth > 20:
            score += 1.0
            growth_good = True
        elif np_growth > 5:
            score += 0.7
            growth_good = True

    if rev_growth is not None:
        if rev_growth > 15:
            score += 0.8
        elif rev_growth > 5:
            score += 0.5

    # C. Sức khỏe tài chính
    if de is not None:
        if de < 0.8:
            score += 1.2
            if len(reasons) < 3:
                reasons.append("Nợ/vốn chủ sở hữu ở mức an toàn")
        elif de < 1.5:
            score += 0.8
        elif de < 2.0:
            score += 0.4

    # D. Định giá hợp lý
    if 0 < pe <= 12:
        score += 1.2
    elif 12 < pe <= 16:
        score += 0.8
    elif 16 < pe <= 20:
        score += 0.4

    if pb is not None:
        if 0 < pb <= 2:
            score += 0.8
        elif pb <= 3:
            score += 0.4

    # E. Vị trí giá tích lũy
    if close is not None and ma50 is not None and close > ma50:
        score += 0.3

    if close is not None and ma100 is not None and close > ma100:
        score += 0.3

    if dist_ma50 is not None and dist_ma50 <= 8:
        score += 0.4
        if len(reasons) < 3:
            reasons.append("Giá vẫn đang ở vùng tích lũy tương đối hợp lý")
    elif dist_ma50 is not None and dist_ma50 <= 12:
        score += 0.2

    if close is not None and ma100 is not None:
        dist_ma100 = abs(close - ma100) / ma100 * 100 if ma100 else None
        if dist_ma100 is not None and dist_ma100 <= 12:
            score += 0.4

    # -------------------------
    # PENALTY
    # -------------------------
    if np_growth is not None and np_growth < 0:
        score -= 0.5

    if rev_growth is not None and rev_growth < 0:
        score -= 0.3

    if pe > 18:
        score -= 0.4

    if de is not None and de > 1.5:
        score -= 0.4

    if dist_ma50 is not None and dist_ma50 > 15:
        score -= 0.5

    if close is not None and ma50 is not None and close < ma50:
        score -= 0.3

    if close is not None and ma100 is not None and close < ma100:
        score -= 0.5

    if x.get("trend_up_mid"):
        score += 0.4

    if x.get("stage2"):
        score += 0.3

    if x.get("regime", 0) > 0:
        score += 0.5

    score = max(0.0, min(10.0, round(score, 1)))

    # -------------------------
    # LABEL + STYLE + RISK
    # -------------------------
    if growth_good and score >= 7.5:
        investment_case = "Tăng trưởng dài hạn đáng chú ý"
        holding_style = "Có thể tích lũy dần cho góc nhìn 1-3 năm nếu tiếp tục giữ chất lượng lợi nhuận"
    elif score >= 6.5:
        investment_case = "Tích lũy dài hạn chất lượng"
        holding_style = "Phù hợp tích lũy từng phần cho trung hạn đến dài hạn"
    else:
        investment_case = "Doanh nghiệp ổn, chờ vùng tích lũy đẹp hơn"
        holding_style = "Nên theo dõi thêm nhịp điều chỉnh hoặc vùng giá hợp lý hơn trước khi tích lũy mạnh"

    risk = "Danh mục dài hạn nên ưu tiên tích lũy từng phần, tránh mua mạnh khi giá đã đi quá xa nền hỗ trợ."

    anchor = ma50 or ma100 or price
    buy_low = round(anchor * 0.96, 2)
    buy_high = round(anchor * 1.03, 2)

    if len(reasons) < 3 and pe is not None:
        reasons.append(f"PE {pe:.1f}, định giá vẫn ở vùng có thể chấp nhận cho tích lũy")
    if len(reasons) < 3 and growth_good:
        reasons.append("Tăng trưởng lợi nhuận vẫn giữ được tín hiệu tích cực")
    if len(reasons) < 3:
        reasons.append("Phù hợp hơn với chiến lược tích lũy dài hạn thay vì mua đuổi ngắn hạn")

    return {
        **x,
        "score": score,
        "investment_case": investment_case,
        "holding_style": holding_style,
        "buy_zone": f"{buy_low} - {buy_high}",
        "reasons": reasons[:3],
        "risk_note": risk,
    }
    
# ============================================================
# FORMATTER
# ============================================================

def _fmt_num(v, digits=2):
    if v is None:
        return "n/a"
    try:
        return f"{float(v):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(v)


def format_weekly_message(penny: List[Dict], short: List[Dict], long_: List[Dict], regime_label: str, market_comment: str) -> str:
    today = datetime.now()
    week_start = today.strftime("%d/%m/%Y")
    week_end = (today + timedelta(days=6)).strftime("%d/%m/%Y")

    def pick_reason(x: Dict, idx: int, fallback: str) -> str:
        reasons = x.get("reasons", [])
        if isinstance(reasons, list) and len(reasons) > idx and reasons[idx]:
            return reasons[idx]
        return fallback

    def render_penny(items: List[Dict]) -> str:
        if not items:
            return "Không có mã phù hợp tuần này."

        lines = []
        for i, x in enumerate(items, 1):
            r1 = pick_reason(
                x, 0,
                "EPS và lợi nhuận đang ở mức chấp nhận được trong nhóm cổ phiếu giá thấp"
            )
            r2 = pick_reason(
                x, 1,
                "Định giá và hiệu quả sử dụng vốn không quá xấu so với mặt bằng cùng nhóm"
            )
            r3 = pick_reason(
                x, 2,
                "Giá đang ở vùng có thể theo dõi tích lũy, phù hợp giải ngân nhỏ từng phần"
            )

            lines.extend([
                f"{i}. {x['ticker']} — Giá: {_fmt_num(x['price'])} — Điểm: {_fmt_num(x['score'], 1)}/10",
                f"   🎯 Nhóm: {x.get('label', 'Theo dõi')}",
                f"   ✅ Lý do:",
                f"   - {r1}",
                f"   - {r2}",
                f"   - {r3}",
                f"   💵 Vùng mua tham khảo: {x.get('buy_zone', 'Theo dõi thêm')}",
                f"   ⚠️ Lưu ý: {x.get('risk_note', 'Nhóm dưới 15.000đ biến động lớn, nên ưu tiên tỷ trọng nhỏ.')}",
                ""
            ])
        return "\n".join(lines).strip()

    def render_short(items: List[Dict]) -> str:
        if not items:
            return "Không có mã phù hợp tuần này."

        lines = []
        for i, x in enumerate(items, 1):
            r1 = pick_reason(
                x, 0,
                "Doanh nghiệp có nền hoạt động ổn và dòng tiền ngắn hạn đang quay lại"
            )
            r2 = pick_reason(
                x, 1,
                "Thanh khoản hiện tại đủ tốt để ưu tiên theo dõi trong danh mục ngắn hạn"
            )
            r3 = pick_reason(
                x, 2,
                "Vị trí giá chưa quá đuổi, phù hợp canh mua theo nhịp điều chỉnh"
            )

            lines.extend([
                f"{i}. {x['ticker']} — Giá: {_fmt_num(x['price'])} — Điểm: {_fmt_num(x['score'], 1)}/10",
                f"   🚀 Setup: {x.get('setup_type', 'Theo dõi ngắn hạn')}",
                f"   ✅ Lý do:",
                f"   - {r1}",
                f"   - {r2}",
                f"   - {r3}",
                f"   💵 Vùng mua đẹp: {x.get('buy_zone', 'Theo dõi thêm')}",
                f"   🎯 Kịch bản: {x.get('entry_plan', 'Ưu tiên mua từng phần khi giá giữ được vùng hỗ trợ gần.')}",
                f"   ⚠️ Hạn chế: {x.get('risk_note', 'Danh mục ngắn hạn cần theo dõi phản ứng giá trong tuần.')}",
                ""
            ])
        return "\n".join(lines).strip()

    def render_long(items: List[Dict]) -> str:
        if not items:
            return "Không có mã phù hợp tuần này."

        lines = []
        for i, x in enumerate(items, 1):
            r1 = pick_reason(
                x, 0,
                "EPS dương, cho thấy doanh nghiệp vẫn đang tạo ra lợi nhuận"
            )
            r2 = pick_reason(
                x, 1,
                "ROE và định giá đang ở mức tương đối phù hợp để theo dõi tích lũy"
            )
            r3 = pick_reason(
                x, 2,
                "Lợi nhuận và vị trí giá hiện tại đủ ổn để cân nhắc cho góc nhìn dài hơn"
            )

            lines.extend([
                f"{i}. {x['ticker']} — Giá: {_fmt_num(x['price'])} — Điểm: {_fmt_num(x['score'], 1)}/10",
                f"   🏦 Luận điểm đầu tư: {x.get('investment_case', 'Tích lũy dài hạn')}",
                f"   ✅ Điểm mạnh:",
                f"   - {r1}",
                f"   - {r2}",
                f"   - {r3}",
                f"   💵 Vùng tích lũy: {x.get('buy_zone', 'Theo dõi thêm')}",
                f"   🕒 Phù hợp: {x.get('holding_style', 'Tích lũy từng phần trong trung hạn đến dài hạn')}",
                f"   ⚠️ Theo dõi thêm: {x.get('risk_note', 'Cần tiếp tục theo dõi thêm kết quả kinh doanh và phản ứng giá.')}",
                ""
            ])
        return "\n".join(lines).strip()

    msg = f"""📊 WEEKLY STOCK WATCHLIST
🗓 Tuần: {week_start} - {week_end}
📈 Market regime: {regime_label}
🔥 Tâm lý thị trường: {market_comment}

━━━━━━━━━━━━━━━━━━
1️⃣ DANH MỤC CP <15.000đ TIỀM NĂNG
━━━━━━━━━━━━━━━━━━
{render_penny(penny)}

━━━━━━━━━━━━━━━━━━
2️⃣ DANH MỤC ƯU TIÊN MUA NGẮN HẠN (<100k)
━━━━━━━━━━━━━━━━━━
{render_short(short)}

━━━━━━━━━━━━━━━━━━
3️⃣ DANH MỤC ƯU TIÊN MUA DÀI HẠN (<100k)
━━━━━━━━━━━━━━━━━━
{render_long(long_)}

━━━━━━━━━━━━━━━━━━
📌 Ghi chú
━━━━━━━━━━━━━━━━━━
- Danh mục mang tính gợi ý tham khảo, không phải khuyến nghị chắc thắng.
- Ưu tiên giải ngân theo vùng mua, không mua đuổi.
- Nhóm <10k nên đi vốn nhỏ hơn 2 nhóm còn lại.
- Danh mục ngắn hạn ưu tiên timing.
- Danh mục dài hạn ưu tiên tích lũy từng phần.
"""
    return msg

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ Thiếu TELEGRAM_TOKEN/TELEGRAM_CHAT_ID -> chỉ in ra console.")
        print(text)
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        chunks = [text[i:i + 3800] for i in range(0, len(text), 3800)]
        for chunk in chunks:
            r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=20)
            if not (r.status_code == 200 and r.json().get("ok")):
                log(f"❌ Telegram lỗi {r.status_code}: {r.text[:300]}")
                return
            time.sleep(0.6)
        log("📨 Đã gửi Telegram.")
    except Exception as e:
        log(f"❌ Telegram error: {e}")


# ============================================================
# MAIN WORKFLOW
# ============================================================

def build_universe_features(tickers: List[str], regime: int) -> List[Dict]:
    rows = []
    total = len(tickers)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = tickers[batch_start: batch_start + BATCH_SIZE]
        batch_no = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        log(f"📦 BATCH {batch_no}/{total_batches} | {len(batch)} mã | vị trí {batch_start + 1}-{batch_start + len(batch)}")

        for j, tk in enumerate(batch, 1):
            global_i = batch_start + j
            log(f"🔎 {global_i}/{total} - {tk}")

            try:
                df = get_price_history(tk, length=PRICE_HISTORY_BARS)
                if df is None or df.empty or len(df) < 120:
                    continue

                fa = get_fa_data(tk)
                row = summarize_features(tk, df, fa, regime)

                if row:
                    rows.append(row)

            except Exception as e:
                log(f"⚠️ lỗi xử lý {tk}: {e}")

        log(f"✅ Xong batch {batch_no}/{total_batches} | usable hiện tại: {len(rows)} mã")

        # Nghỉ giữa batch, trừ batch cuối
        if batch_start + BATCH_SIZE < total:
            log(f"😴 Nghỉ {BATCH_DELAY_SEC}s trước batch tiếp theo để tránh nghẽn API...")
            time.sleep(BATCH_DELAY_SEC)

    log(f"✅ Build feature xong: {len(rows)} mã usable")
    return rows


def rank_bucket(items: List[Optional[Dict]], top_n: int = TOP_N_PER_BUCKET) -> List[Dict]:
    clean = [x for x in items if x]
    clean.sort(key=lambda x: x.get("score", 0), reverse=True)
    return clean[:top_n]



def run_scan():
    configure_vnstock_auth()

    tickers = get_tickers_from_sheet()
    if not tickers:
        log("❌ Không có tickers từ Google Sheet.")
        return

    regime, regime_label, market_comment = get_market_regime()
    log(f"📈 Market regime = {regime_label}")

    rows = build_universe_features(tickers, regime)
    if not rows:
        log("❌ Không build được universe features.")
        return

    penny = []
    short = []
    long_ = []

    penny_near = []
    long_near = []

    for x in rows:
        p = score_penny(x)
        if p:
            penny.append(p)
        else:
            # near-pass cho nhóm <10k
            price = x.get("price")
            value20 = x.get("value20") or 0
            eps = x.get("eps")
            latest_np = x.get("latest_net_profit")
            roe = x.get("roe")
            pe = x.get("pe")

            if price and price < PENNY_MAX_PRICE and value20 >= 1e9:
                near_score = 0.0
                reasons = []

                if eps is not None and eps > 0:
                    near_score += 1.0
                    reasons.append(f"EPS dương ({eps:.0f}), nền lợi nhuận không quá yếu")
                if latest_np is not None and latest_np > 0:
                    near_score += 1.0
                    if len(reasons) < 3:
                        reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")
                if roe is not None and roe >= 3:
                    near_score += 0.8
                    if len(reasons) < 3:
                        reasons.append(f"ROE {roe:.1f}% ở mức chấp nhận được")
                if pe is not None and 0 < pe < 18:
                    near_score += 0.7
                    if len(reasons) < 3:
                        reasons.append(f"PE {pe:.1f}, định giá chưa quá cao")
                if value20 >= 2e9:
                    near_score += 0.8
                    if len(reasons) < 3:
                        reasons.append("Thanh khoản đủ để tiếp tục theo dõi")

                if len(reasons) < 3:
                    reasons.append("Phù hợp giữ trong danh sách theo dõi hơn là giải ngân mạnh")

                penny_near.append({
                    **x,
                    "score": round(near_score, 1),
                    "label": "Theo dõi thêm",
                    "buy_zone": f"{round(price * 0.96, 2)} - {round(price * 1.04, 2)}",
                    "reasons": reasons[:3],
                    "risk_note": "Chưa đạt đầy đủ tiêu chuẩn ưu tiên, chỉ nên theo dõi hoặc giải ngân rất nhỏ.",
                })

        s = score_short_term(x)
        if s:
            short.append(s)

        l = score_long_term(x)
        if l:
            long_.append(l)
        else:
            # near-pass cho nhóm dài hạn
            price = x.get("price")
            value20 = x.get("value20") or 0
            eps = x.get("eps")
            latest_np = x.get("latest_net_profit")
            roe = x.get("roe")
            pe = x.get("pe")
            de = x.get("de")
            ma50 = x.get("ma50") or price

            if price and price < LONG_MAX_PRICE and value20 >= 1e9 and (
                (eps is not None and eps > 0) or (latest_np is not None and latest_np > 0)
            ):
                near_score = 0.0
                reasons = []

                if eps is not None and eps > 0:
                    near_score += 1.0
                    reasons.append(f"EPS dương ({eps:.0f}), doanh nghiệp vẫn tạo ra lợi nhuận")
                if latest_np is not None and latest_np > 0:
                    near_score += 1.0
                    if len(reasons) < 3:
                        reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")
                if roe is not None and roe >= 7:
                    near_score += 0.8
                    if len(reasons) < 3:
                        reasons.append(f"ROE {roe:.1f}% ở mức khá")
                if pe is not None and 0 < pe < 22:
                    near_score += 0.8
                    if len(reasons) < 3:
                        reasons.append(f"PE {pe:.1f}, định giá tương đối hợp lý")
                if de is not None and de < 2:
                    near_score += 0.5
                    if len(reasons) < 3:
                        reasons.append("Nợ/vốn chủ sở hữu chưa ở mức đáng lo")
                if value20 >= 2e9:
                    near_score += 0.6

                if len(reasons) < 3:
                    reasons.append("Có thể giữ trong danh sách theo dõi để chờ điểm mua đẹp hơn")

                long_near.append({
                    **x,
                    "score": round(near_score, 1),
                    "investment_case": "Doanh nghiệp tạm ổn, theo dõi để tích lũy",
                    "holding_style": "Theo dõi thêm trước khi tích lũy mạnh",
                    "buy_zone": f"{round(ma50 * 0.97, 2)} - {round(ma50 * 1.03, 2)}",
                    "reasons": reasons[:3],
                    "risk_note": "Chưa phải lựa chọn dài hạn mạnh, phù hợp theo dõi thêm hơn là giải ngân lớn ngay.",
                })

    penny = sorted([x for x in penny if x], key=lambda x: x["score"], reverse=True)
    short = sorted([x for x in short if x], key=lambda x: x["score"], reverse=True)
    long_ = sorted([x for x in long_ if x], key=lambda x: x["score"], reverse=True)

    if len(penny) < TOP_N_PER_BUCKET:
        penny_near = sorted(penny_near, key=lambda x: x["score"], reverse=True)
        need = TOP_N_PER_BUCKET - len(penny)
        penny.extend(penny_near[:need])

    if len(long_) < TOP_N_PER_BUCKET:
        long_near = sorted(long_near, key=lambda x: x["score"], reverse=True)
        need = TOP_N_PER_BUCKET - len(long_)
        long_.extend(long_near[:need])

    penny = penny[:TOP_N_PER_BUCKET]
    short = short[:TOP_N_PER_BUCKET]
    long_ = long_[:TOP_N_PER_BUCKET]

    msg = format_weekly_message(penny, short, long_, regime_label, market_comment)
    print(msg)
    send_telegram(msg)


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "scan").lower().strip()
    log(f"🚀 Start mode={mode}")

    if mode == "scan":
        run_scan()
    elif mode == "print":
        run_scan()
    elif mode == "clear_cache":
        for p in [PRICE_CACHE_FILE, FA_CACHE_FILE, UNIVERSE_CACHE_FILE]:
            if os.path.exists(p):
                os.remove(p)
                log(f"🗑 Xóa cache {os.path.basename(p)}")
    else:
        log("❌ Mode không hỗ trợ. Dùng: python main.py scan")


if __name__ == "__main__":
    main()
