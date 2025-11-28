# ===========================
#  VN STOCK BOT: FA = vnstock, TA = TCBS
#  - B1: Lấy danh sách mã từ Google Sheet (cp < 10k bạn đã lọc sẵn)
#  - B2: FA từ vnstock (VCI source) -> cache 7 ngày
#  - B3: TA từ TCBS (OHLC daily) -> mỗi lần scan
# ===========================

import os, sys, json, time
from datetime import datetime, timedelta
import requests
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import ta
import csv
from vnstock import Vnstock, Quote


# ---------- ENV & CACHE DIR ----------
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
FA_CACHE_FILE = os.path.join(CACHE_DIR, "fa_cache.json")
SEASONALITY_FILE = os.path.join(CACHE_DIR, "seasonality_cache.json")

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------- HTTP SESSION ----------
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "vnstock-bot/1.0",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    })
    retry = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    s.mount("http://",  HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    return s

SESSION = make_session()

# ---------- SIMPLE CACHE ----------
def cache_get(name, ttl_sec):
    p = os.path.join(CACHE_DIR, name)
    try:
        if os.path.exists(p) and (time.time() - os.path.getmtime(p) < ttl_sec):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def cache_set(name, obj):
    p = os.path.join(CACHE_DIR, name)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass
def get_monthly_returns_vnstock(ticker, years=10):
    """
    Lấy dữ liệu 1D ~10 năm bằng vnstock Quote.history,
    tính lợi nhuận trung bình theo tháng (1..12)
    """
    try:
        tk = str(ticker).upper().strip()
        quote = Quote(symbol=tk, source='VCI')   # hoặc 'TCBS' cũng được

        end = datetime.now().date()
        start = end - timedelta(days=365 * years)

        df = quote.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval='1D'
        )
        if df is None or df.empty:
            return {}

        # cột thời gian trong vnstock là 'time'
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month

        grp = df.groupby(["year", "month"])
        monthly_ret = (grp["close"].last() / grp["close"].first() - 1.0).reset_index()
        if monthly_ret.empty:
            return {}

        avg_by_month = monthly_ret.groupby("month")["close"].mean().to_dict()
        return avg_by_month

    except Exception as e:
        log(f"⚠️ Seasonality {ticker} lỗi: {e}")
        return {}



def rebuild_seasonality_cache(tickers):
    """
    Phân tích seasonality cho list tickers, lưu cache:
    {
      "built_key": "YYYY-MM",
      "good_months": {
         "CII": [2,3,4],
         "HPG": [2,3],
         ...
      }
    }
    Tháng "tốt" = các tháng có return dương và thuộc top 4 tháng cao nhất của mã.
    """
    if not tickers:
        return {}

    log(f"📊 Rebuild seasonality cache cho {len(tickers)} mã …")
    good_months = {}
    for i, tk in enumerate(tickers, 1):
        tk = str(tk).upper().strip()
        log(f"   ↳ Seasonality {i}/{len(tickers)} – {tk}")
        avg_month = get_monthly_returns_vnstock(tk, years=10)
        if not avg_month:
            continue

        # Lọc tháng dương & top 4
        items = [(m, r) for m, r in avg_month.items() if r > 0]
        if not items:
            continue
        # sort theo return desc
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:4]   # lấy tối đa 4 tháng "đỉnh" nhất
        good_months[tk] = [m for m, _ in top]

    built_key = datetime.now().strftime("%Y-%m")
    data = {"built_key": built_key, "good_months": good_months}
    cache_set(SEASONALITY_FILE, data)
    log(f"✅ Seasonality cache xong: {len(good_months)} mã.")
    return good_months


def load_seasonality_cache():
    data = cache_get(SEASONALITY_FILE, ttl_sec=365*24*3600)  # TTL 1 năm
    if not data:
        return None, {}
    return data.get("built_key"), data.get("good_months", {})


def ensure_seasonality(tickers):
    """
    Đảm bảo tháng này đã có seasonality cache.
    Nếu chưa có hoặc khác tháng hiện tại -> rebuild.
    Trả về: dict {ticker -> [good_months]}
    """
    current_key = datetime.now().strftime("%Y-%m")
    built_key, good_months = load_seasonality_cache()
    if built_key == current_key and good_months:
        log(f"🟢 Seasonality cache sẵn có cho {len(good_months)} mã (built={built_key}).")
        return good_months

    log("🟡 Seasonality cache chưa có / khác tháng -> rebuild…")
    return rebuild_seasonality_cache(tickers)

# ============================================================
# B1) LẤY DANH SÁCH MÃ TỪ GOOGLE SHEET (bạn đã lọc <10k ở đó)
# ============================================================

def get_tickers_from_sheet():
    if not SHEET_CSV_URL:
        log("⚠️ SHEET_CSV_URL chưa cấu hình.")
        return []

    try:
        df = pd.read_csv(SHEET_CSV_URL)
    except Exception as e:
        log(f"❌ Lỗi đọc sheet: {e}")
        return []

    # tìm cột mã: "Mã", "ma", "ticker", "symbol", "code"
    col_ticker = None
    for c in df.columns:
        if str(c).strip().lower() in ["mã", "ma", "ticker", "symbol", "code"]:
            col_ticker = c
            break
    if col_ticker is None:
        # fallback cột A
        col_ticker = df.columns[0]

    tks = (
        df[col_ticker]
        .astype(str).str.upper().str.strip()
        .dropna().unique().tolist()
    )
    tks = sorted(set([tk for tk in tks if tk and tk != "NAN"]))
    log(f"✅ Sheet lấy được {len(tks)} mã cp (đã lọc sẵn).")
    return tks

# ============================================================
# B2) FA TỪ VNSTOCK (SOURCE = VCI) + CACHE 7 NGÀY
# ============================================================

def _find_col(df: pd.DataFrame, keywords):
    """
    Tìm tên cột chứa 1 trong các keyword (bỏ khoảng trắng, lowercase).
    Dùng để dò 'EPS', 'ROE', 'P/E', 'Debt/Equity'… trong bảng ratio vnstock.
    """
    kws = [k.lower().replace(" ", "") for k in keywords]
    for col in df.columns:
        key = str(col).lower().replace(" ", "").replace("_", "")
        for k in kws:
            if k in key:
                return col
    return None

def get_fa_one_ticker_vnstock(tk: str):
    """
    Lấy FA cho 1 mã từ vnstock (VCI source) – dùng bảng ratio (year).
    Trả về dict với: ticker, eps, roe, pe, de (debt/equity)
    hoặc None nếu lỗi / thiếu dữ liệu.
    """
    symbol = tk.upper().strip()
    try:
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        ratio_df = stock.finance.ratio(period="year", lang="vi", dropna=True)
        if ratio_df is None or ratio_df.empty:
            log(f"🟡 FA vnstock rỗng cho {symbol}")
            return None

        # lấy dòng mới nhất (thường là năm gần nhất)
        row = ratio_df.iloc[-1]

        col_eps = _find_col(ratio_df, ["eps"])
        col_roe = _find_col(ratio_df, ["roe"])
        col_pe  = _find_col(ratio_df, ["p/e", "pe"])
        col_de  = _find_col(ratio_df, ["nợ/vốn", "debttoequity", "debt/equity", "d/e"])

        def _get(row, col):
            if col is None:
                return None
            try:
                v = row[col]
                return float(v) if pd.notna(v) else None
            except Exception:
                return None

        eps = _get(row, col_eps)
        roe = _get(row, col_roe)
        pe  = _get(row, col_pe)
        de  = _get(row, col_de)

        if eps is None or roe is None or pe is None:
            log(f"🟡 Thiếu cột FA (EPS/ROE/PE) cho {symbol}")
            return None

        return {
            "ticker": symbol,
            "eps": eps,
            "roe": roe,
            "pe": pe,
            "de": de,
        }

    except Exception as e:
        log(f"⚠️ FA vnstock lỗi {symbol}: {e}")
        return None

def run_fa_update_vnstock(tickers):
    """
    Tải FA cho list tickers từ vnstock và lưu vào cache 7 ngày: fa_cache.json
    """
    if not tickers:
        log("❌ Không có tickers để cập nhật FA.")
        return pd.DataFrame()

    log(f"🧾 Cập nhật FA (vnstock/VCI) cho {len(tickers)} mã …")
    rows = []
    for i, tk in enumerate(tickers, 1):
        fa = get_fa_one_ticker_vnstock(tk)
        if fa:
            fa["fa_growth_score"] = calc_fa_growth_score(tk)
            rows.append(fa)

        if i % 20 == 0:
            log(f"…đã lấy FA {i}/{len(tickers)} mã")
        time.sleep(0.2)

    df = pd.DataFrame(rows)
    cache_set("fa_cache.json", {"rows": rows, "ts": int(time.time())})
    log(f"✅ Lưu cache FA (vnstock): {len(df)} mã (7 ngày).")
    return df

def load_fa_cache():
    cached = cache_get("fa_cache.json", ttl_sec=7 * 24 * 3600)
    if cached and cached.get("rows"):
        return pd.DataFrame(cached["rows"])
    return pd.DataFrame()

def analyze_fa(df: pd.DataFrame):
    """
    FA filter (đơn giản hơn bản VNDIRECT cũ):
      - EPS > 300
      - ROE > 8 (%)
      - 0 < PE < 15
      - Debt/Equity < 1.5 (nếu có)
    """
    if df is None or df.empty:
        return []

    fa_pass = []
    for _, r in df.iterrows():
        try:
            tk  = str(r["ticker"]).upper()
            eps = float(r.get("eps", 0) or 0)
            roe = float(r.get("roe", 0) or 0)
            pe  = float(r.get("pe",  0) or 0)
            de  = r.get("de", None)
            de  = float(de) if de is not None else None
        except Exception:
            continue

        if eps <= 300:
            continue
        if roe <= 8:
            continue
        if not (0 < pe < 15):
            continue
        if de is not None and de >= 1.5:
            continue

        fa_pass.append({
            "ticker": tk,
            "eps": eps,
            "roe": roe,
            "pe": pe,
            "de": de,
        })

    log(f"✅ FA PASS (vnstock): {len(fa_pass)} mã")
    return fa_pass
def calc_fa_growth_score(ticker: str, years: int = 5):
    try:
        stock = Vnstock().stock(symbol=ticker, source="VCI")
        df = stock.finance.income_statement(period="year", dropna=True)
        if df is None or df.empty: return 0

        df = df.tail(years)

        col_np = None
        for c in df.columns:
            k = str(c).lower().replace(" ", "")
            if "netprofit" in k or "lnst" in k or "loinhuansauthue" in k:
                col_np = c; break
        if col_np is None: return 0

        lst = df[col_np].astype(float).dropna().tolist()
        if len(lst) < 3: return 0

        last = lst[-3:]

        score = 0
        if all(x > 0 for x in last): score += 1
        if last[0] > 0:
            cagr = (last[-1]/last[0])**(1/(len(last)-1)) - 1
            if cagr > 0.05: score += 1

        return score
    except:
        return 0

# ============================================================
# B3) TA TỪ TCBS (OHLC DAILY)
# ============================================================

def get_ohlc_days_tcbs(tk: str, days: int = 180):
    """
    Lấy nến ngày từ TCBS:
      https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term
    """
    ticker = tk.upper().strip()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=int(days * 1.2))

    start_ts = int(time.mktime(datetime.combine(start_date, datetime.min.time()).timetuple()))
    end_ts   = int(time.mktime(datetime.combine(end_date,   datetime.min.time()).timetuple()))

    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",
        "from": start_ts,
        "to": end_ts,
    }

    try:
        r = SESSION.get(url, params=params, timeout=(8, 20))
        if r.status_code == 404:
            log(f"⛔ TCBS không có dữ liệu cho {ticker}, bỏ qua.")
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            log(f"🟡 TCBS trả rỗng cho {ticker}.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "tradingDate" not in df.columns:
            log(f"🟡 TCBS thiếu cột tradingDate cho {ticker}.")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["tradingDate"].str.split("T", expand=True)[0]).dt.date

        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = pd.NA

        df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(df) > days:
            df = df.iloc[-days:].reset_index(drop=True)
        return df

    except Exception as e:
        log(f"⚠️ OHLC TCBS lỗi {ticker}: {e}")
        return pd.DataFrame()

def technical_signals(df: pd.DataFrame):
    """
    5 điều kiện TA:
      - ADX > 20 & DI+ > DI-
      - RSI > 50 và vừa cắt lên
      - Break đỉnh 20 phiên
      - Volume tăng 3 phiên liên tiếp
      - Close > MA20 & Volume Spike
    """
    conds = {}
    if df is None or len(df) < 25:
        conds["enough_data"] = False
        conds["score_TA_true"] = 0
        return conds, 0

    rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_ind.rsi()
    adx_ind = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx_ind.adx()
    df["di_pos"] = adx_ind.adx_pos()
    df["di_neg"] = adx_ind.adx_neg()
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]; prev = df.iloc[-2]
    conds["ADX>20_DI+>DI-"]   = bool((latest["adx"] > 20) and (latest["di_pos"] > latest["di_neg"]))
    conds["RSI>50_cross_up"]  = bool((latest["rsi"] > 50) and (prev["rsi"] <= 50))
    conds["Break_20_high"]    = bool(latest["close"] > float(df["close"].iloc[-20:-1].max()))
    conds["Vol_up_3_days"]    = bool(df["volume"].iloc[-1] > df["volume"].iloc[-2] > df["volume"].iloc[-3])
    conds["Close>MA20_VolSp"] = bool((latest["close"] > latest["ma20"]) and (latest["volume"] > 1.5 * latest["vol_ma20"]))

    score = sum(1 for v in conds.values() if v)
    conds["enough_data"] = True
    conds["score_TA_true"] = score
    return conds, score
def calc_buy_tp(df):
    """
    Buy zone = MA20 ± 3%
    TP zone  = Fibonacci extension 1.618 – 2.0 từ Swing High
    """
    if df is None or len(df) < 30:
        return None, None

    latest = df.iloc[-1]
    ma20 = latest["ma20"]

    if pd.isna(ma20) or ma20 <= 0:
        return None, None

    # --- Buy zone ---
    buy_low  = round(ma20 * 0.97)
    buy_high = round(ma20 * 1.03)

    # --- TP zone (Fibonacci extension) ---
    # Swing High = đỉnh cao nhất 20 phiên gần nhất
    swing_high = max(df["close"].iloc[-20:])

    tp_low  = round(swing_high * 1.618)  # Fibo 161.8%
    tp_high = round(swing_high * 2.0)    # Fibo 200%

    return (buy_low, buy_high), (tp_low, tp_high)
# ========================
#  EXTRA: NEAR BUY + LIQ
# ========================
LIQ_VALUE_MIN = 3e9   # giá trị giao dịch TB 20 phiên tối thiểu (3 tỷ)

def calc_near_buy_and_liquidity(df):
    """
    - near_buy_bonus:
        +2 nếu |price - MA20| < 3%
        +1 nếu < 6%
        +0 nếu xa hơn
    - liquidity:
        loại nếu GTGD TB20 < LIQ_VALUE_MIN
        +1 nếu >= 2 * LIQ_VALUE_MIN
    - stage2_bonus:
        +1 nếu Close > MA20 > MA50 > MA100
        0 nếu không

    return:
        (ok:bool, near_buy_bonus:int, liq_bonus:int, stage2_bonus:int)
    """
    if df is None or len(df) < 60:
        return False, 0, 0, 0

    close = float(df["close"].iloc[-1])

    # MA20 / 50 / 100
    if "ma20" not in df.columns:
        df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"]  = df["close"].rolling(50).mean()
    df["ma100"] = df["close"].rolling(100).mean()

    ma20  = float(df["ma20"].iloc[-1])
    ma50  = float(df["ma50"].iloc[-1])
    ma100 = float(df["ma100"].iloc[-1])

    if any(pd.isna(x) or x <= 0 for x in [close, ma20, ma50, ma100]):
        return False, 0, 0, 0

    # khoảng cách tới MA20
    dist = abs(close - ma20) / ma20
    if dist < 0.03:
        near_bonus = 2
    elif dist < 0.06:
        near_bonus = 1
    else:
        near_bonus = 0

    # thanh khoản: GTGD TB20
    if "volume" not in df.columns or df["volume"].isna().all():
        return False, 0, 0, 0

    value = df["close"] * df["volume"]
    value20 = float(value.rolling(20).mean().iloc[-1])

    if pd.isna(value20) or value20 < LIQ_VALUE_MIN:
        return False, 0, 0, 0

    if value20 >= 2 * LIQ_VALUE_MIN:
        liq_bonus = 1
    else:
        liq_bonus = 0

    # Stage 2: Close > MA20 > MA50 > MA100
    stage2_bonus = 1 if (close > ma20 > ma50 > ma100) else 0

    return True, near_bonus, liq_bonus, stage2_bonus


# ============================================================
# TELEGRAM FORMAT & SEND
# ============================================================

def send_telegram(text):
    token = TELEGRAM_TOKEN; chat = TELEGRAM_CHAT_ID
    if not token or not chat:
        log("❌ Thiếu TELEGRAM_TOKEN / TELEGRAM_CHAT_ID"); return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat, "text": text}, timeout=15)
        if r.status_code == 200 and r.json().get("ok"):
            log("📨 Sent Telegram.")
        else:
            log(f"❌ Telegram {r.status_code}: {r.text}")
    except Exception as e:
        log(f"❌ Telegram error: {e}")

def format_msg_fa_ta(stocks):
    """
    Mỗi mã 1 dòng:
    MÃ; BUY_LOW-BUY_HIGH; TP_LOW-TP_HIGH
    (Áp dụng cho các mã đạt FA + TA)
    """
    today = datetime.now().strftime("%d/%m/%Y")

    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt FA + TA (≥3/5)."

    lines = []
    for s in stocks:
        tk    = s["ticker"]
        buy   = s.get("buy_zone")
        tp    = s.get("tp_zone")
        score = s.get("ta_score", "?")
        star  = "⭐️ " if s.get("season") else ""
    
        if buy and tp:
            lines.append(f"{star}{tk}; {buy[0]}-{buy[1]}; {tp[0]}-{tp[1]}; TA:{score}/5")
    

    msg = f"💹 [{today}] Mã <30k đạt FA + TA (≥3/5):\n" + "\n".join(lines)
    return msg


def format_msg_ta_only(stocks):
    """
    Mỗi mã 1 dòng:
    MÃ; BUY_LOW-BUY_HIGH; TP_LOW-TP_HIGH
    """
    today = datetime.now().strftime("%d/%m/%Y")

    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt TA (≥3/5)."

    lines = []
    for s in stocks:
        tk    = s["ticker"]
        buy   = s.get("buy_zone")
        tp    = s.get("tp_zone")
        score = s.get("ta_score", "?")
        star  = "⭐️ " if s.get("season") else ""
    
        if buy and tp:
            lines.append(f"{star}{tk}; {buy[0]}-{buy[1]}; {tp[0]}-{tp[1]}; TA:{score}/5")



    msg = f"📈 [{today}] Mã <30k đạt TA (≥3/5):\n" + "\n".join(lines)
    return msg

def log_signals_to_csv(stocks, mode_label: str):
    """
    Ghi log tín hiệu vào signals_log.csv để backtest sau này.
    mode_label: 'TA_ONLY' hoặc 'FA_TA'
    """
    if not stocks:
        return

    log_path = os.path.join(BASE_DIR, "signals_log.csv")
    file_exists = os.path.exists(log_path)

    fieldnames = [
        "date", "mode", "ticker",
        "buy_low", "buy_high",
        "tp_low", "tp_high",
        "ta_score", "fa_growth_score",
        "near_buy_bonus", "liq_bonus", "stage2_bonus",
        "season", "total_score"
    ]

    today = datetime.now().strftime("%Y-%m-%d")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for s in stocks:
            buy = s.get("buy_zone")
            tp  = s.get("tp_zone")
            if not (buy and tp):
                continue

            row = {
                "date": today,
                "mode": mode_label,
                "ticker": s.get("ticker"),
                "buy_low":  buy[0],
                "buy_high": buy[1],
                "tp_low":   tp[0],
                "tp_high":  tp[1],
                "ta_score": s.get("ta_score"),
                "fa_growth_score": s.get("fa_growth_score"),
                "near_buy_bonus": s.get("near_buy_bonus"),
                "liq_bonus": s.get("liq_bonus"),
                "stage2_bonus": s.get("stage2_bonus"),
                "season": int(bool(s.get("season"))),
                "total_score": s.get("total_score"),
            }
            writer.writerow(row)

    log(f"📝 Đã ghi {len(stocks)} tín hiệu vào signals_log.csv ({mode_label}).")

# ============================================================
# MAIN
#   python main.py list  -> chỉ lấy danh sách mã từ sheet
#   python main.py fa    -> cập nhật & cache FA từ vnstock
#   python main.py scan  -> FA cache (nếu có) + TA realtime
# ============================================================

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "scan").lower()
    log(f"🚀 Start BOT mode={mode}")

    tks = get_tickers_from_sheet()
    if not tks:
        log("⚠️ Không lấy được danh sách mã từ Sheet.")
        return
    # Seasonality: build cache 1 lần / tháng
    season_map = ensure_seasonality(tks)  # dict {ticker: [good_months]}
    current_month = datetime.now().month

    # ==== FA AUTO: chỉ update FA lúc 19h Thứ 6 VN ====
    now_utc = datetime.utcnow()
    now_vn  = now_utc + timedelta(hours=7)
    if now_vn.weekday() == 4 and now_vn.hour == 19:   # 4 = Friday
        log("🔄 Thứ 6 19h VN → CẬP NHẬT FA (vnstock)…")
        run_fa_update_vnstock(tks)
    else:
        log("⏭ Không phải 19h Thứ 6 → dùng FA cache cũ, không update.")
    # ==== MODE = FA: cho phép bạn tự chạy bằng tay (python main.py fa) ====
    if mode == "fa":
        log("🔄 MODE=fa → Cập nhật FA (vnstock) theo yêu cầu …")
        run_fa_update_vnstock(tks)
        log("⚡ FA Update DONE.")
        return
    # ==== MODE = SCAN (mặc định) — chỉ đọc cache FA ====
    df_fa_cache = load_fa_cache()
    fa_list = analyze_fa(df_fa_cache) if not df_fa_cache.empty else []

    if not fa_list:
        log("🟠 Không dùng được FA (cache rỗng hoặc không mã nào pass) → TA-only.")
        # === TA-ONLY ===
        final = []
        for i, tk in enumerate(tks, 1):
            log(f"[TA-only] {i}/{len(tks)} – {tk}")
            df = get_ohlc_days_tcbs(tk, days=180)
            if df.empty:
                continue
        
            conds, score = technical_signals(df)
            if not (conds.get("enough_data") and score >= 3):
                continue
        
            # Tính BUY/TP
            buy_zone, tp_zone = calc_buy_tp(df)
            if not (buy_zone and tp_zone):
                continue
            # Tính near_buy_bonus + liquidity filter
            ok_liq, near_bonus, liq_bonus, stage2_bonus = calc_near_buy_and_liquidity(df)

            if not ok_liq:
                continue    
            is_season = False
            if season_map and tk in season_map:
                if current_month in season_map[tk]:
                    is_season = True
        
            fa_growth = 0
            base_total = (
                score
                + near_bonus
                + liq_bonus
                + stage2_bonus
                + (1 if is_season else 0)
                + fa_growth * 0.5       # BONUS cho FA growth
            )

        
            final.append({
                "ticker": tk,
                "ta_score": score,
                "buy_zone": buy_zone,
                "tp_zone": tp_zone,
                "near_buy_bonus": near_bonus,
                "stage2_bonus": stage2_bonus,
                "liq_bonus": liq_bonus,
                "season": is_season,
                "fa_growth_score": fa_growth,
                "total_score": base_total,
            })

            time.sleep(0.15)
        # sort giảm dần theo total_score
        final.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        log_signals_to_csv(final, mode_label="TA_ONLY")
        send_telegram(format_msg_ta_only(final))
        log(f"ALL DONE (TA-only). Final={len(final)}")
        return

    # Nếu FA có dữ liệu thì chạy FA -> TA
    # === FA + TA ===
    final = []
    for i, it in enumerate(fa_list, 1):
        tk = it["ticker"]
        log(f"[FA+TA] {i}/{len(fa_list)} — {tk}")
        df = get_ohlc_days_tcbs(tk, days=180)
        if df.empty:
            continue
    
        conds, score = technical_signals(df)
        if not (conds.get("enough_data") and score >= 3):
            continue
    
        buy_zone, tp_zone = calc_buy_tp(df)
        if not (buy_zone and tp_zone):
            continue
    
        ok_liq, near_bonus, liq_bonus, stage2_bonus = calc_near_buy_and_liquidity(df)

        if not ok_liq:
            continue
    
        is_season = False
        if season_map and tk in season_map:
            if current_month in season_map[tk]:
                is_season = True
    
        fa_growth = it.get("fa_growth_score", 0) or 0
        base_total = (
            score
            + near_bonus
            + liq_bonus
            + stage2_bonus
            + (1 if is_season else 0)
            + fa_growth * 0.5       # BONUS cho FA growth
        )

    
        final.append({
            **it,
            "ta_score": score,
            "buy_zone": buy_zone,
            "tp_zone": tp_zone,
            "near_buy_bonus": near_bonus,
            "liq_bonus": liq_bonus,
            "season": is_season,
            "stage2_bonus": stage2_bonus,
            "fa_growth_score": fa_growth,
            "total_score": base_total,
        })

        time.sleep(0.15)
    # sort giảm dần theo total_score
    final.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    log_signals_to_csv(final, mode_label="TA_ONLY")
    send_telegram(format_msg_fa_ta(final))
    log(f"ALL DONE (FA+TA). Final={len(final)}")

if __name__ == "__main__":
    main()
