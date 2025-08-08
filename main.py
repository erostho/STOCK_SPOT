import os
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from dotenv import load_dotenv

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
def get_fundamentals_latest_quarters(size=4000):
    log("📥 Lấy BCTC từ VNDIRECT...")
    params = {
        "q": "reportType:QUARTER",
        "size": size,
        "sort": "ticker,-yearQuarter"
    }
    try:
        res = requests.get(FR_URL, params=params, timeout=25)
        res.raise_for_status()
        data = res.json().get("data", [])
        df = pd.DataFrame(data)
        log(f"✅ Lấy {len(df)} dòng BCTC")
        return df
    except Exception as e:
        log(f"❌ Lỗi lấy BCTC: {e}")
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
        eps   = fcol(latest, "eps", 0.0)
        pe    = fcol(latest, "pe", 0.0)
        roe   = fcol(latest, "roe", 0.0)

        lnst_quarters = pd.to_numeric(sub.get("netProfit"), errors="coerce").fillna(0.0).values.tolist()
        cfo_quarters = pd.to_numeric(sub.get("netCashFlowFromOperatingActivities"), errors="coerce").fillna(0.0).values.tolist()

        inventory_latest = fcol(latest, "inventory", 0.0)
        liabilities_latest = fcol(latest, "liabilities", 0.0)
        equity_latest = fcol(latest, "equity", 0.0)
        inv_yoy = fcol(sub.iloc[4].to_dict(), "inventory", None) if len(sub) >= 5 else None

        log(f"🔍 FA {ticker}: Giá={price:.0f} EPS={eps:.0f} ROE={roe:.1f} P/E={pe:.1f}")

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
def technical_signals(df):
    if df.empty or len(df) < 25:
        return {}, 0

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    rsi = ta.rsi(close, length=14)
    adx_df = ta.adx(high=df["high"], low=df["low"], close=close, length=14)
    ma20 = ta.sma(close, length=20)
    vol_ma20 = ta.sma(vol, length=20)

    conds = {}
    adx_val = float(adx_df.iloc[-1]["ADX_14"])
    dip_val = float(adx_df.iloc[-1]["DMP_14"])
    dim_val = float(adx_df.iloc[-1]["DMN_14"])
    conds["ADX>20_DI+>DI-"] = (adx_val > 20) and (dip_val > dim_val)

    rsi_curr = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2])
    conds["RSI>50_cross_up"] = (rsi_curr > 50) and (rsi_prev <= 50)

    last_close = float(close.iloc[-1])
    last_20_high = float(close.iloc[-20:-1].max())
    conds["Break_20_high"] = last_close > last_20_high

    conds["Vol_up_3_days"] = (vol.iloc[-1] > vol.iloc[-2] > vol.iloc[-3])
    conds["Close>MA20_VolSpike"] = (last_close > float(ma20.iloc[-1])) and \
                                   (float(vol.iloc[-1]) > 1.5 * float(vol_ma20.iloc[-1]))

    count_true = sum(1 for v in conds.values() if v)
    conds["score_TA_true"] = count_true
    return conds, count_true

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
