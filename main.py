
import os
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ====== API VNDIRECT ======
FINFO_BASE = "https://finfo-api.vndirect.com.vn/v4"
FR_URL = f"{FINFO_BASE}/financial_reports"
PRICE_URL = f"{FINFO_BASE}/stock_prices"

# ====== LOG ======
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

# ====== 1. Lấy BCTC 4 quý ======
# session dùng chung có retry
def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

SESSION = make_session()

def get_fundamentals_latest_quarters(size=1500):
    log("📥 Lấy BCTC từ VNDIRECT (có retry/backoff)...")
    params = {
        "q": "reportType:QUARTER",
        "size": size,                 # nhỏ hơn để tránh time-out
        "sort": "ticker,-yearQuarter"
    }
    try:
        res = SESSION.get(FR_URL, params=params, timeout=40)
        res.raise_for_status()
        data = res.json().get("data", [])
        df = pd.DataFrame(data)
        log(f"✅ Lấy {len(df)} dòng BCTC")
        return df
    except Exception as e:
        log(f"❌ Lỗi lấy BCTC (đã retry): {e}")
        return pd.DataFrame()

# ====== 2. Phân tích FA nâng cao ======
def analyze_fa_multi(df_quarter):
    if df_quarter.empty:
        return []

    fa_pass = []
    for ticker, sub in df_quarter.groupby("ticker"):
        sub = sub.sort_values(by="yearQuarter", ascending=False).head(8)

        def fcol(row, key, default=0.0):
            try:
                val = row.get(key, default)
                return float(val) if pd.notna(val) else default
            except:
                return default

        latest = sub.iloc[0].to_dict()
        price = fcol(latest, "price", 0.0)
        eps = fcol(latest, "eps", 0.0)
        pe = fcol(latest, "pe", 0.0)
        roe = fcol(latest, "roe", 0.0)

        lnst_quarters = pd.to_numeric(sub.get("netProfit"), errors="coerce").fillna(0.0).values.tolist()
        cfo_quarters = pd.to_numeric(sub.get("netCashFlowFromOperatingActivities"), errors="coerce").fillna(0.0).values.tolist()

        inventory_latest = fcol(latest, "inventory", 0.0)
        liabilities_latest = fcol(latest, "liabilities", 0.0)
        equity_latest = fcol(latest, "equity", 0.0)
        inv_yoy = fcol(sub.iloc[4].to_dict(), "inventory", None) if len(sub) >= 5 else None

        if not (price > 0 and price < 10000):
            continue
        if eps <= 500 or roe <= 10 or pe <= 0 or pe >= 10:
            continue
        if equity_latest <= 0 or liabilities_latest / equity_latest >= 1.0:
            continue
        if sum(cfo_quarters[:4]) <= 0:
            continue

        lnst_yoy_ok = False
        if len(lnst_quarters) >= 5:
            if lnst_quarters[0] > lnst_quarters[4]:
                lnst_yoy_ok = True
        elif len(lnst_quarters) >= 8:
            if sum(lnst_quarters[0:4]) > sum(lnst_quarters[4:8]):
                lnst_yoy_ok = True
        if not lnst_yoy_ok:
            continue

        if inv_yoy and inv_yoy > 0:
            inv_growth = (inventory_latest - inv_yoy) / inv_yoy
            if inv_growth > 0.30:
                continue

        log("   ✅ FA đạt")
        fa_pass.append({
            "ticker": ticker,
            "price": price,
            "eps": eps,
            "roe": roe,
            "pe": pe
        })

    log(f"✅ FA PASS: {len(fa_pass)} mã")
    return fa_pass

# ====== 3. Lấy nến OHLC ======
def get_ohlc_days(ticker, days=120):
    start = (datetime.now() - timedelta(days=days*2)).strftime("%Y-%m-%d")
    params = {
        "q": f"ticker:{ticker}~date:gte:{start}",
        "sort": "date",
        "size": 1000
    }
    try:
        res = requests.get(PRICE_URL, params=params, timeout=20)
        res.raise_for_status()
        data = res.json().get("data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        for c, newc in [("adOpen","open"),("adHigh","high"),("adLow","low"),("adClose","close"),("adVolume","volume")]:
            if c in df.columns:
                df[newc] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date","open","high","low","close","volume"]].dropna()
        df = df.sort_values("date").reset_index(drop=True)
        return df.tail(150)
    except Exception as e:
        log(f"❌ Lỗi lấy nến {ticker}: {e}")
        return pd.DataFrame()

# ====== 4. Phân tích TA ======
# dùng lib 'ta' thay cho pandas_ta

def technical_signals(df: pd.DataFrame):
    """
    Trả về (conds, score):
      conds: dict các điều kiện TA (True/False)
      score: số điều kiện đạt (>=3 là pass theo multi-confirmation)

    Điều kiện:
      - ADX > 20 và DI+ > DI-
      - RSI > 50 và vừa cắt lên (rsi[-1] > 50 & rsi[-2] <= 50)
      - Break đỉnh 20 phiên (close[-1] > max(close[-20:-1]))
      - Volume tăng 3 phiên liên tiếp
      - Close > MA20 và Volume Spike (vol[-1] > 1.5 * vol_ma20[-1])
    """
    conds = {}

    # Kiểm tra dữ liệu đủ dài
    if df is None or len(df) < 25:
        conds["enough_data"] = False
        conds["score_TA_true"] = 0
        return conds, 0

    # ----- Chỉ báo với thư viện 'ta'
    # RSI(14)
    rsi_ind = ta.momentum.RSIIndicator(close=df["close"], window=14)
    df["rsi"] = rsi_ind.rsi()

    # ADX(14), DI+ / DI-
    adx_ind = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx_ind.adx()
    df["di_pos"] = adx_ind.adx_pos()
    df["di_neg"] = adx_ind.adx_neg()

    # MA20 & Vol MA20 (dùng rolling pandas cho nhẹ)
    df["ma20"] = df["close"].rolling(20).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    # ----- 5 điều kiện TA
    # 1) ADX > 20 & DI+ > DI-
    conds["ADX>20_DI+>DI-"] = bool((latest["adx"] > 20) and (latest["di_pos"] > latest["di_neg"]))

    # 2) RSI > 50 và cắt lên (cross up 50)
    conds["RSI>50_cross_up"] = bool((latest["rsi"] > 50) and (prev["rsi"] <= 50))

    # 3) Break đỉnh 20 phiên
    last_20_high = float(df["close"].iloc[-20:-1].max())
    conds["Break_20_high"] = bool(latest["close"] > last_20_high)

    # 4) Volume tăng 3 phiên liên tiếp
    try:
        conds["Vol_up_3_days"] = bool(
            df["volume"].iloc[-1] > df["volume"].iloc[-2] > df["volume"].iloc[-3]
        )
    except Exception:
        conds["Vol_up_3_days"] = False

    # 5) Close > MA20 & Volume Spike
    conds["Close>MA20_VolSpike"] = bool(
        (latest["close"] > latest["ma20"]) and (latest["volume"] > 1.5 * latest["vol_ma20"])
    )

    # Tính score
    score = sum(1 for k, v in conds.items() if v)
    conds["enough_data"] = True
    conds["score_TA_true"] = score

    return conds, score

# ====== 5. Multi-confirmation ======
def multi_confirmation_filter(fa_list, ta_min=3):
    final = []
    for item in fa_list:
        ticker = item["ticker"]
        log(f"📊 TA cho {ticker}...")
        df = get_ohlc_days(ticker, days=180)
        conds, score = technical_signals(df)
        if not conds:
            continue
        log(f"   TA {ticker}: {conds} | Score={score}")
        if score >= ta_min:
            final.append({**item, "ta_score": score})
    log(f"✅ MULTI PASS: {len(final)} mã")
    return final

# ====== 6. Gửi Telegram ======
def format_telegram_message(stocks):
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [BOT CỔ PHIẾU] {today}\nKhông mã nào đạt FA + TA (multi-confirmation)."
    msg = f"📈 [BOT CỔ PHIẾU] {today}\nMã <10.000đ đạt FA + TA (≥3/5):\n\n"
    for s in stocks:
        msg += (f"🔹 {s['ticker']} | Giá: {int(s['price'])}đ | EPS: {int(s['eps'])} | "
                f"ROE: {s['roe']:.1f}% | P/E: {s['pe']:.1f} | TA✓: {s['ta_score']}/5\n")
    return msg

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code == 200:
            log("📨 Đã gửi kết quả Telegram.")
        else:
            log(f"❌ Telegram lỗi: {r.status_code} | {r.text}")
    except Exception as e:
        log(f"❌ Lỗi gửi Telegram: {e}")

# ====== 7. Main ======
def main():
    log("🚀 KHỞI ĐỘNG BOT — Multi-Confirmation FA+TA")
    df_quarter = get_fundamentals_latest_quarters()
    fa_list = analyze_fa_multi(df_quarter)
    final = multi_confirmation_filter(fa_list, ta_min=3)
    msg = format_telegram_message(final)
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
