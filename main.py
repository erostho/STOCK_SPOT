# ===========================
#  BOT LỌC CỔ PHIẾU VIỆT NAM
#  FA + TA (Multi-confirmation)
#  Nguồn chính: VNDIRECT finfo-api
#  Tối ưu: retry/backoff, cache, per-ticker
#  Tác giả: bạn & trợ lý :)
# ===========================

# =====[1] IMPORTS & ENV =====
import os, json, time
import requests
import pandas as pd
import ta  # thư viện chỉ báo kỹ thuật (không dùng pandas_ta)
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
# Cho phép dùng 2 tên biến: TELEGRAM_TOKEN hoặc TELEGRAM_BOT_TOKEN
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

FINFO_BASE = "https://finfo-api.vndirect.com.vn/v4"
FR_URL     = f"{FINFO_BASE}/financial_reports"
PRICE_URL  = f"{FINFO_BASE}/stock_prices"

# In log có flush để Render hiển thị tức thời
def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

# =====[2] HTTP SESSION (retry/backoff) =====
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": "vnstock-bot/1.0",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=6, connect=6, read=6,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    s.mount("http://",  HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    return s

SESSION = make_session()

# =====[3] CACHE ĐƠN GIẢN (24h) =====
CACHE_DIR = "/tmp/vnstock_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_get(name: str, ttl_sec: int):
    path = os.path.join(CACHE_DIR, name)
    try:
        if os.path.exists(path) and (time.time() - os.path.getmtime(path) < ttl_sec):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def cache_set(name: str, obj):
    path = os.path.join(CACHE_DIR, name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
    except Exception:
        pass

# =====[4] LẤY DANH SÁCH <10K TỪ LATEST FA (NHẸ + CACHE) =====
def get_tickers_under_10k():
    # dùng cache 24h nếu có
    cached = cache_get("tickers_under_10k.json", ttl_sec=24 * 3600)
    if cached:
        tks = cached.get("tickers", [])
        log(f"✅ Dùng cache tickers: {len(tks)} mã")
        return tks

    log("📥 Lọc mã <10k từ financial_reports (~isLatest:true)…")
    try:
        params = {"q": "reportType:QUARTER~isLatest:true", "size": 1500, "sort": "ticker"}
        r = SESSION.get(FR_URL, params=params, timeout=(60, 120))
        r.raise_for_status()
        df = pd.DataFrame(r.json().get("data", []))
        if df.empty:
            raise RuntimeError("latest FA rỗng")

        price = pd.to_numeric(df.get("price"), errors="coerce")
        tickers = (
            df.loc[(price > 0) & (price < 10000), "ticker"]
              .dropna().unique().tolist()
        )
        tickers = sorted(tickers)
        log(f"✅ Ứng viên <10k: {len(tickers)} mã. (đã cache 24h)")
        cache_set("tickers_under_10k.json", {"tickers": tickers})
        return tickers
    except Exception as e:
        log(f"🟠 Lỗi lấy latest FA: {e} → thử dùng cache (nếu có)")
        return cached.get("tickers", []) if cached else []

# =====[5] BCTC: LẤY THEO TỪNG MÃ (NHỎ, ÍT TIMEOUT) =====
def get_fr_one_ticker(tk: str):
    try:
        params = {"q": f"ticker:{tk}~reportType:QUARTER", "size": 8, "sort": "-yearQuarter"}
        r = SESSION.get(FR_URL, params=params, timeout=(40, 80))
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        log(f"⚠️ {tk} FR lỗi: {e}")
        return []

def get_fundamentals_latest_quarters():
    log("📥 Tải BCTC 8 quý / mã…")
    tickers = get_tickers_under_10k()
    out = []
    for i, tk in enumerate(tickers, 1):
        out.extend(get_fr_one_ticker(tk))
        if i % 20 == 0:
            log(f"…đã lấy {i}/{len(tickers)} mã BCTC")
            time.sleep(0.3)
    df = pd.DataFrame(out)
    log(f"✅ Tổng dòng BCTC: {len(df)}")
    return df

# =====[6] PHÂN TÍCH FA NÂNG CAO (BẮT BUỘC) =====
def analyze_fa_multi(df_quarter: pd.DataFrame):
    """
    Điều kiện FA bắt buộc:
      - Giá < 10,000
      - EPS > 500
      - ROE > 10%
      - 0 < PE < 10
      - Nợ/Vốn < 1
      - CFO (TTM) dương
      - LNST tăng trưởng YoY (ưu tiên cùng kỳ; fallback TTM)
      - Tồn kho YoY không tăng > 30%
    """
    if df_quarter.empty:
        return []

    fa_pass = []
    for ticker, sub in df_quarter.groupby("ticker"):
        sub = sub.sort_values(by="yearQuarter", ascending=False).head(8)
        latest = sub.iloc[0].to_dict()

        def f(row, key, default=0.0):
            try:
                v = row.get(key, default)
                return float(v) if pd.notna(v) else default
            except Exception:
                return default

        price = f(latest, "price")
        eps   = f(latest, "eps")
        pe    = f(latest, "pe")
        roe   = f(latest, "roe")
        inventory_latest  = f(latest, "inventory")
        liabilities_latest= f(latest, "liabilities")
        equity_latest     = f(latest, "equity")

        lnst_quarters = pd.to_numeric(sub.get("netProfit"), errors="coerce").fillna(0.0).values.tolist()
        cfo_quarters  = pd.to_numeric(sub.get("netCashFlowFromOperatingActivities"), errors="coerce").fillna(0.0).values.tolist()

        inv_yoy = None
        if len(sub) >= 5:
            inv_yoy = f(sub.iloc[4].to_dict(), "inventory", None)

        # Checks
        if not (0 < price < 10000):           continue
        if not (eps > 500):                    continue
        if not (roe > 10):                     continue
        if not (0 < pe < 10):                  continue
        if equity_latest <= 0 or (liabilities_latest / equity_latest) >= 1.0:
            continue
        cfo_ttm = sum(cfo_quarters[:4]) if len(cfo_quarters) >= 4 else sum(cfo_quarters)
        if cfo_ttm <= 0:                       continue

        lnst_yoy_ok = False
        if len(lnst_quarters) >= 5:
            lnst_yoy_ok = lnst_quarters[0] > lnst_quarters[4]
        elif len(lnst_quarters) >= 8:
            lnst_yoy_ok = sum(lnst_quarters[0:4]) > sum(lnst_quarters[4:8])
        if not lnst_yoy_ok:                    continue

        if inv_yoy and inv_yoy > 0:
            inv_growth = (inventory_latest - inv_yoy) / inv_yoy
            if inv_growth > 0.30:              continue

        fa_pass.append({
            "ticker": ticker,
            "price": price,
            "eps": eps,
            "roe": roe,
            "pe": pe
        })
    log(f"✅ FA PASS: {len(fa_pass)} mã")
    return fa_pass

# =====[7] OHLC: LẤY 120 PHIÊN / MÃ =====
def get_ohlc_days(ticker: str, days: int = 120) -> pd.DataFrame:
    start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")
    params = {"q": f"ticker:{ticker}~date:gte:{start}", "sort": "date", "size": 1000}
    try:
        r = SESSION.get(PRICE_URL, params=params, timeout=(40, 80))
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        for c, n in [("adOpen","open"),("adHigh","high"),("adLow","low"),("adClose","close"),("adVolume","volume")]:
            if c in df.columns:
                df[n] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date","open","high","low","close","volume"]].dropna().sort_values("date")
        return df.tail(150)
    except Exception as e:
        log(f"⚠️ OHLC {ticker} lỗi: {e}")
        return pd.DataFrame()

# =====[8] TA: CHỈ BÁO & MULTI-CONFIRMATION (>=3/5) =====
def technical_signals(df: pd.DataFrame):
    """
    5 điều kiện TA:
      - ADX > 20 & DI+ > DI-
      - RSI > 50 và vừa cắt lên
      - Break đỉnh 20 phiên (close[-1] > max(close[-20:-1]))
      - Volume tăng 3 phiên liên tiếp
      - Close > MA20 & Volume Spike (vol > 1.5 * vol_ma20)
    """
    conds = {}
    if df is None or len(df) < 25:
        conds["enough_data"] = False
        conds["score_TA_true"] = 0
        return conds, 0

    # RSI
    rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_ind.rsi()

    # ADX & DI
    adx_ind = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"]    = adx_ind.adx()
    df["di_pos"] = adx_ind.adx_pos()
    df["di_neg"] = adx_ind.adx_neg()

    # MA20 & Vol MA20
    df["ma20"]     = df["close"].rolling(20).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    conds["ADX>20_DI+>DI-"]   = bool((latest["adx"] > 20) and (latest["di_pos"] > latest["di_neg"]))
    conds["RSI>50_cross_up"]  = bool((latest["rsi"] > 50) and (prev["rsi"] <= 50))
    conds["Break_20_high"]    = bool(latest["close"] > float(df["close"].iloc[-20:-1].max()))
    conds["Vol_up_3_days"]    = bool(df["volume"].iloc[-1] > df["volume"].iloc[-2] > df["volume"].iloc[-3])
    conds["Close>MA20_VolSp"] = bool((latest["close"] > latest["ma20"]) and (latest["volume"] > 1.5 * latest["vol_ma20"]))

    score = sum(1 for v in conds.values() if v)
    conds["enough_data"] = True
    conds["score_TA_true"] = score
    return conds, score

def multi_confirmation_filter(fa_list, ta_min=3):
    final = []
    for item in fa_list:
        tk = item["ticker"]
        log(f"📊 TA cho {tk}…")
        df = get_ohlc_days(tk, days=180)
        conds, score = technical_signals(df)
        if not conds.get("enough_data", False):
            log("   ⚠️ TA: dữ liệu không đủ")
            continue
        log(f"   TA {tk}: {conds} | Score={score}")
        if score >= ta_min:
            final.append({**item, "ta_score": score})
    log(f"✅ MULTI PASS: {len(final)} mã")
    return final

# =====[9] TELEGRAM =====
def format_telegram_message(stocks):
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [BOT CỔ PHIẾU] {today}\nKhông mã nào đạt FA + TA (multi-confirmation)."
    msg = f"📈 [BOT CỔ PHIẾU] {today}\nMã <10.000đ đạt FA bắt buộc + TA (≥3/5):\n\n"
    for s in stocks:
        msg += (f"🔹 {s['ticker']} | Giá: {int(s['price'])}đ | EPS: {int(s['eps'])} | "
                f"ROE: {s['roe']:.1f}% | P/E: {s['pe']:.1f} | TA✓: {s['ta_score']}/5\n")
    msg += "\n#vnstock #fa #ta #multi_confirmation"
    return msg

def send_telegram_message(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("❌ Thiếu TELEGRAM_TOKEN/TELEGRAM_CHAT_ID")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=20)
        if r.status_code == 200 and r.json().get("ok"):
            log("📨 Đã gửi kết quả Telegram.")
        else:
            log(f"❌ Telegram lỗi {r.status_code}: {r.text}")
    except Exception as e:
        log(f"❌ Lỗi gửi Telegram: {e}")

# =====[10] MAIN (CHECKPOINT LOG) =====
def main():
    log("🚀 KHỞI ĐỘNG BOT — Multi-Confirmation FA+TA")

    log("STEP 1: Lấy BCTC từng mã")
    df_quarter = get_fundamentals_latest_quarters()
    log(f"STEP 1 DONE: rows={len(df_quarter)}")

    if df_quarter.empty:
        msg = "⚠️ BOT: Không lấy được dữ liệu BCTC hôm nay (VNDIRECT chậm). Sẽ thử lại lần sau."
        log(msg); send_telegram_message(msg); return

    log("STEP 2: Phân tích FA")
    fa_list = analyze_fa_multi(df_quarter)
    log(f"STEP 2 DONE: FA pass={len(fa_list)}")

    log("STEP 3: Phân tích TA (multi-confirmation)")
    final = multi_confirmation_filter(fa_list, ta_min=3)
    log(f"STEP 3 DONE: final={len(final)}")

    log("STEP 4: Gửi Telegram")
    msg = format_telegram_message(final)
    send_telegram_message(msg)
    log("ALL DONE ✅")

if __name__ == "__main__":
    main()
