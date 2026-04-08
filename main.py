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
RATE_LIMIT_PER_MIN = int(os.getenv("VNSTOCK_RATE_LIMIT_PER_MIN", "55"))  # giữ đệm dưới 60
REQUEST_TIMESTAMPS = deque()

# ---------- user knobs ----------
TOP_N_PER_BUCKET = 5
PRICE_HISTORY_BARS = 320
PRICE_TTL_SEC = 24 * 3600
FA_TTL_SEC = 7 * 24 * 3600
UNIVERSE_TTL_SEC = 24 * 3600

PENNY_MAX_PRICE = 10_000
SHORT_MAX_PRICE = 50_000
LONG_MAX_PRICE = 50_000

PENNY_LIQ_MIN = 3e9
SHORT_LIQ_MIN = 7e9
LONG_LIQ_MIN = 3e9


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

    result = {
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
    }

    try:
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        finance = stock.finance

        ratio_df = _vns_call(finance.ratio, period="year", lang="vi", dropna=True)
        if ratio_df is not None and not ratio_df.empty:
            row = ratio_df.iloc[-1]
            result["eps"] = _safe_float(row.get(_find_col(ratio_df, ["eps"])))
            result["roe"] = _safe_float(row.get(_find_col(ratio_df, ["roe"])))
            result["pe"] = _safe_float(row.get(_find_col(ratio_df, ["p/e", "pe"])))
            result["pb"] = _safe_float(row.get(_find_col(ratio_df, ["p/b", "pb"])))
            result["de"] = _safe_float(row.get(_find_col(ratio_df, ["nợ/vốn", "debttoequity", "debt/equity", "d/e"])))

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

        # FA quality score nhẹ, dùng cho ranking
        score = 0
        if result["eps"] and result["eps"] > 0:
            score += 1
        if result["roe"] and result["roe"] >= 10:
            score += 1
        if result["pe"] and 0 < result["pe"] < 18:
            score += 1
        if result["de"] is not None and result["de"] < 1.5:
            score += 1
        if result["np_growth"] is not None and result["np_growth"] > 0:
            score += 1
        result["fa_quality_score"] = score

    except Exception as e:
        log(f"⚠️ FA lỗi {ticker}: {e}")

    cache_all[ticker] = result
    cache_save(FA_CACHE_FILE, cache_all)
    return result


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
    dist_ma20 = x.get("dist_ma20")
    close = x.get("close")
    ma20 = x.get("ma20")
    vol_ratio = x.get("vol_ratio") or 0

    # 1) Thanh khoản: nới từ 3 tỷ xuống 2 tỷ
    if value20 < 2e9:
        return None

    # 2) ROE: nếu có thì chỉ cần >= 3
    if roe is not None and roe < 3:
        return None

    # 3) D/E: nếu có thì cho rộng hơn
    if de is not None and de >= 3:
        return None

    # 4) FA mềm hơn: chỉ cần 1 trong 3 điều kiện là được
    fa_ok = False
    if eps is not None and eps > 0:
        fa_ok = True
    if latest_np is not None and latest_np > 0:
        fa_ok = True
    if np_growth is not None and np_growth > 0:
        fa_ok = True
    if not fa_ok:
        return None

    # 5) Kỹ thuật: chỉ cần không quá xấu
    tech_ok = False
    if dist_ma20 is not None and dist_ma20 <= 12:
        tech_ok = True
    if close is not None and ma20 is not None and close >= ma20 * 0.95:
        tech_ok = True
    if not tech_ok:
        return None

    # 6) Không lấy mã đã tăng quá xa nền ngắn hạn
    if dist_ma20 is not None and dist_ma20 > 18:
        return None

    score = 0.0
    reasons = []

    # EPS
    if eps is not None and eps > 0:
        score += 1.0
        reasons.append(f"EPS dương ({eps:.0f}), nền lợi nhuận không quá yếu")

    # ROE
    if roe is not None:
        if roe >= 8:
            score += 1.2
            reasons.append(f"ROE {roe:.1f}% ở mức khá trong nhóm cổ phiếu giá thấp")
        elif roe >= 5:
            score += 0.8
            reasons.append(f"ROE {roe:.1f}% ở mức chấp nhận được")
        elif roe >= 3:
            score += 0.4

    # PE
    if pe is not None:
        if 0 < pe < 12:
            score += 1.0
            reasons.append(f"PE {pe:.1f}, định giá chưa quá cao")
        elif 0 < pe < 18:
            score += 0.5

    # Lợi nhuận / tăng trưởng
    if latest_np is not None and latest_np > 0:
        score += 1.0
        reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")

    if np_growth is not None:
        if np_growth > 10:
            score += 1.2
            reasons.append("Lợi nhuận đang cải thiện khá rõ")
        elif np_growth > 0:
            score += 0.7
            if len(reasons) < 3:
                reasons.append("Lợi nhuận có dấu hiệu cải thiện")

    # Thanh khoản
    if value20 >= 5e9:
        score += 1.0
        if len(reasons) < 3:
            reasons.append("Thanh khoản khá tốt so với mặt bằng nhóm dưới 10.000đ")
    elif value20 >= 3e9:
        score += 0.7
    else:
        score += 0.4

    # Vị trí giá / kỹ thuật chỉ là phụ
    if dist_ma20 is not None:
        if dist_ma20 <= 6:
            score += 1.0
            if len(reasons) < 3:
                reasons.append("Giá đang ở gần vùng nền hỗ trợ, phù hợp theo dõi tích lũy")
        elif dist_ma20 <= 12:
            score += 0.5

    if x.get("trend_up_short"):
        score += 0.8
    if x.get("breakout20"):
        score += 0.6
    if x.get("stage2"):
        score += 0.5
    if vol_ratio >= 1.2:
        score += 0.6
    if x.get("regime", 0) > 0:
        score += 0.5

    score = min(10.0, round(score, 1))

    # Label
    if np_growth is not None and np_growth > 0:
        label = "Hồi phục đáng chú ý"
        risk = "Nhóm dưới 10.000đ biến động lớn, nên giải ngân nhỏ và chia lệnh."
    elif x.get("breakout20") or (dist_ma20 is not None and dist_ma20 <= 6):
        label = "Tích lũy chờ breakout"
        risk = "Phù hợp mua thăm dò quanh vùng hỗ trợ, tránh mua đuổi khi tăng nóng."
    else:
        label = "Thăm dò sớm"
        risk = "Ưu tiên theo dõi và giải ngân nhỏ, không phù hợp dồn vốn mạnh."

    buy_low = round(max(0, price * 0.97), 2)
    buy_high = round(price * 1.03, 2)

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
    if x["value20"] < SHORT_LIQ_MIN:
        return None
    if not x["close"] > (x.get("ma20") or 0):
        return None
    if not (x["trend_up_short"] or x["pullback_uptrend"] or x["retest_breakout"]):
        return None
    if not x["rsi_ok"]:
        return None
    if (x.get("dist_ma20") or 999) > 8:
        return None

    score = 0.0
    reasons = []
    setup_type = "Theo dõi thêm"
    entry_plan = "Canh vùng hỗ trợ gần, tránh mua đuổi lúc nến bốc mạnh."

    if x["breakout20"] or x["breakout60"]:
        score += 2.0
        reasons.append("Đang có tín hiệu breakout khỏi nền tích lũy")
        setup_type = "Breakout nền tích lũy"
        entry_plan = "Ưu tiên chờ breakout rõ hoặc retest thành công rồi mua từng phần."
    if x["retest_breakout"]:
        score += 2.0
        reasons.append("Đang retest sau breakout, điểm mua dễ chịu hơn")
        setup_type = "Retest sau breakout"
        entry_plan = "Canh retest ổn định gần nền breakout, không mua đuổi xa nền."
    if x["pullback_uptrend"]:
        score += 1.8
        reasons.append("Nhịp hồi trong xu hướng tăng vẫn còn khỏe")
        if setup_type == "Theo dõi thêm":
            setup_type = "Pullback trong uptrend"
            entry_plan = "Có thể mua từng phần quanh MA20 nếu lực bán không tăng mạnh."
    if x["adx_ok"]:
        score += 1.3
        reasons.append("Động lượng xu hướng đang ủng hộ chiều tăng")
    if x["vol_ratio"] and x["vol_ratio"] >= 1.25:
        score += 1.0
        reasons.append("Khối lượng đang cao hơn trung bình")
    if x["trend_up_mid"]:
        score += 1.0
    if x["near_ma20_bonus"] >= 1:
        score += 0.8
    if x.get("fa_quality_score", 0) >= 2:
        score += 0.6
    if x["regime"] > 0:
        score += 0.5

    score = min(10.0, round(score, 1))

    if score >= 8.0:
        risk = "Setup khá đẹp nhưng vẫn nên giải ngân theo từng phần, không mua đuổi nến xanh mạnh."
    else:
        risk = "Đà tăng có nhưng cần quan sát thêm phản ứng quanh hỗ trợ gần."

    anchor = x["ma20"] if x["pullback_uptrend"] else x["price"]
    buy_low = round(anchor * 0.985, 2)
    buy_high = round(anchor * 1.015, 2)

    return {
        **x,
        "score": score,
        "setup_type": setup_type,
        "buy_zone": f"{buy_low} - {buy_high}",
        "entry_plan": entry_plan,
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
    de = x.get("de")
    latest_np = x.get("latest_net_profit")
    np_growth = x.get("np_growth")
    dist_ma50 = x.get("dist_ma50")
    close = x.get("close")
    ma50 = x.get("ma50")

    # 1) Thanh khoản: nới từ 3 tỷ xuống 2 tỷ
    if value20 < 2e9:
        return None

    # 2) Chỉ giữ EPS dương là điều kiện cứng
    if eps is None or eps <= 0:
        return None

    # 3) Các chỉ số khác: nếu có thì mới lọc
    if roe is not None and roe < 7:
        return None

    if pe is not None and not (0 < pe < 22):
        return None

    if de is not None and de >= 2:
        return None

    # 4) Vị trí giá không quá xấu là được
    tech_ok = False
    if dist_ma50 is not None and dist_ma50 <= 12:
        tech_ok = True
    if close is not None and ma50 is not None and close >= ma50 * 0.93:
        tech_ok = True
    if not tech_ok:
        return None

    score = 0.0
    reasons = []
    investment_case = "Doanh nghiệp ổn định để tích lũy"
    holding_style = "Tích lũy 3-6 tháng hoặc dài hơn"

    # EPS
    score += 1.0
    reasons.append(f"EPS dương ({eps:.0f}), doanh nghiệp vẫn tạo ra lợi nhuận")

    # ROE
    if roe is not None:
        if roe >= 15:
            score += 1.5
            reasons.append(f"ROE {roe:.1f}% ở mức tốt")
        elif roe >= 10:
            score += 1.0
            reasons.append(f"ROE {roe:.1f}% ở mức khá")
        elif roe >= 7:
            score += 0.5

    # PE
    if pe is not None:
        if 0 < pe <= 15:
            score += 1.2
            reasons.append(f"PE {pe:.1f}, định giá tương đối hợp lý")
        elif pe < 22:
            score += 0.6

    # D/E
    if de is not None:
        if de < 1:
            score += 1.0
            reasons.append("Nợ/vốn chủ sở hữu ở mức an toàn")
        elif de < 2:
            score += 0.5

    # Lợi nhuận / tăng trưởng
    if latest_np is not None and latest_np > 0:
        score += 1.0
        if len(reasons) < 3:
            reasons.append("Lợi nhuận gần nhất vẫn duy trì dương")

    if np_growth is not None:
        if np_growth > 10:
            score += 1.2
            investment_case = "Tăng trưởng + định giá hợp lý"
            if len(reasons) < 3:
                reasons.append("Tăng trưởng lợi nhuận đang tích cực")
        elif np_growth > 0:
            score += 0.7
            if len(reasons) < 3:
                reasons.append("Lợi nhuận vẫn giữ được tăng trưởng dương")

    # Thanh khoản
    if value20 >= 5e9:
        score += 1.0
    elif value20 >= 3e9:
        score += 0.7
    else:
        score += 0.4

    # Vị trí tích lũy
    if dist_ma50 is not None:
        if dist_ma50 <= 6:
            score += 1.0
            if len(reasons) < 3:
                reasons.append("Giá đang ở vùng tích lũy tương đối thuận lợi")
        elif dist_ma50 <= 12:
            score += 0.5

    # Kỹ thuật phụ
    if x.get("trend_up_mid"):
        score += 0.8
    if x.get("stage2"):
        score += 0.6
    if x.get("regime", 0) > 0:
        score += 0.5

    score = min(10.0, round(score, 1))

    if x.get("breakout20") and (dist_ma50 is not None and dist_ma50 > 8):
        holding_style = "Doanh nghiệp tốt nhưng nên chờ nhịp điều chỉnh bớt nóng"
        risk = "FA ổn nhưng điểm mua hiện tại chưa thật sự đẹp nếu giá đã kéo xa hỗ trợ."
    else:
        risk = "Phù hợp tích lũy từng phần, ưu tiên mua gần vùng hỗ trợ trung hạn."

    anchor = ma50 if ma50 else price
    buy_low = round(anchor * 0.97, 2)
    buy_high = round(anchor * 1.03, 2)

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
                f"   ⚠️ Lưu ý: {x.get('risk_note', 'Nhóm dưới 10.000đ biến động lớn, nên ưu tiên tỷ trọng nhỏ.')}",
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
1️⃣ DANH MỤC CP <10.000đ TIỀM NĂNG
━━━━━━━━━━━━━━━━━━
{render_penny(penny)}

━━━━━━━━━━━━━━━━━━
2️⃣ DANH MỤC ƯU TIÊN MUA NGẮN HẠN (<50k)
━━━━━━━━━━━━━━━━━━
{render_short(short)}

━━━━━━━━━━━━━━━━━━
3️⃣ DANH MỤC ƯU TIÊN MUA DÀI HẠN (<50k)
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
    for i, tk in enumerate(tickers, 1):
        log(f"🔎 {i}/{len(tickers)} - {tk}")
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

    penny = rank_bucket([score_penny(x) for x in rows], TOP_N_PER_BUCKET)
    short = rank_bucket([score_short_term(x) for x in rows], TOP_N_PER_BUCKET)
    long_ = rank_bucket([score_long_term(x) for x in rows], TOP_N_PER_BUCKET)

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
