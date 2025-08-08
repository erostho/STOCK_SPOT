# ===========================
#  VN STOCK BOT: FireAnt + VNDIRECT
#  B1: FireAnt -> tickers < 10k (nhẹ)
#  B2: VNDIRECT -> FA (cache 7 ngày)
#  B3: VNDIRECT -> TA (realtime mỗi lần scan)
# ===========================

import os, json, time, sys
import requests
import pandas as pd
import ta
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------- ENV ----------
FINFO_BASE = "https://finfo-api.vndirect.com.vn/v4"
FR_URL     = f"{FINFO_BASE}/financial_reports"
PRICE_URL  = f"{FINFO_BASE}/stock_prices"

# FireAnt free: thay endpoint/token theo thực tế của bạn
FIREANT_BASE = os.getenv("FIREANT_BASE", "https://api.fireant.vn")  # ví dụ
FIREANT_TOKEN = (os.getenv("FIREANT_TOKEN") or "").strip()

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

CACHE_DIR = "/tmp/vnstock_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------- HTTP SESSION ----------
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
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"]
    )
    s.mount("https://", HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    s.mount("http://",  HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retry))
    return s

SESSION = make_session()

# ---------- CACHE ----------
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
            json.dump(obj, f)
    except Exception:
        pass

# ============================================================
# B1) LẤY DANH SÁCH <10K TỪ FIREANT (NHẸ)
# ============================================================
def get_tickers_under_10k_from_fireant():
    """
    Lấy toàn thị trường từ FireAnt (endpoint ví dụ) -> lọc price < 10,000.
    Bạn cần chỉnh endpoint/params theo token & tài liệu FireAnt free bạn có.
    """
    log("📥 FireAnt: lấy danh sách & lọc <10k … (retry ngắn)")
    # ví dụ endpoint: /symbols or /prices (tùy FireAnt free của bạn)
    url = f"{FIREANT_BASE}/symbols"
    headers = {"Authorization": f"Bearer {FIREANT_TOKEN}"} if FIREANT_TOKEN else {}
    params = {"type": "stock"}  # tuỳ endpoint
    last_err = None
    for attempt in range(1, 4):
        try:
            r = SESSION.get(url, headers=headers, params=params, timeout=(8, 18))
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame(rows)
            # Chuẩn hoá: cố gắng lấy cột giá (tuỳ API: 'price', 'lastPrice', 'refPrice'…)
            price = None
            for col in ["price", "lastPrice", "matchPrice", "close", "refPrice"]:
                if col in df.columns:
                    price = pd.to_numeric(df[col], errors="coerce")
                    break
            if price is None:
                raise RuntimeError("Không tìm thấy cột giá trong phản hồi FireAnt")

            tickers = (
                df.loc[(price > 0) & (price < 10000), "symbol"].dropna()
                  .astype(str).str.upper().unique().tolist()
                if "symbol" in df.columns
                else df.loc[(price > 0) & (price < 10000), "ticker"].dropna()
                     .astype(str).str.upper().unique().tolist()
            )
            tickers = sorted(tickers)
            log(f"✅ FireAnt <10k: {len(tickers)} mã.")
            cache_set("tickers_under_10k.json", {"tickers": tickers, "src": "fireant"})
            return tickers
        except Exception as e:
            last_err = e
            log(f"⚠️ FireAnt attempt {attempt}/3: {e}")
            time.sleep(1.2)

    # Fallback: dùng cache cũ nếu có
    cached = cache_get("tickers_under_10k.json", ttl_sec=24*3600)
    if cached and cached.get("tickers"):
        log(f"🟡 FireAnt lỗi, dùng cache: {len(cached['tickers'])} mã")
        return cached["tickers"]
    log(f"❌ FireAnt không khả dụng: {last_err}")
    return []

# ============================================================
# B2) FA TỪ VNDIRECT (CÓ CACHE 7 NGÀY) - CHẠY RIÊNG
# ============================================================
def get_fr_one_ticker_vnd(tk):
    try:
        params = {"q": f"ticker:{tk}~reportType:QUARTER", "size": 8, "sort": "-yearQuarter"}
        r = SESSION.get(FR_URL, params=params, timeout=(8, 18))
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        log(f"⚠️ {tk} FR lỗi: {e}")
        return []

def run_fa_update(tickers):
    """Tải FA cho list tickers và lưu cache 7 ngày: fa_cache.json"""
    if not tickers:
        log("❌ Không có tickers để cập nhật FA.")
        return []
    log(f"🧾 Cập nhật FA cho {len(tickers)} mã …")
    out = []
    for i, tk in enumerate(tickers, 1):
        out.extend(get_fr_one_ticker_vnd(tk))
        if i % 25 == 0:
            log(f"…đã lấy {i}/{len(tickers)} mã FA")
            time.sleep(0.3)
    df = pd.DataFrame(out)
    cache_set("fa_cache.json", {"rows": out, "ts": int(time.time())})
    log(f"✅ Lưu cache FA: {len(df)} dòng (7 ngày)")
    return df

def load_fa_cache():
    cached = cache_get("fa_cache.json", ttl_sec=7*24*3600)
    if cached and cached.get("rows"):
        return pd.DataFrame(cached["rows"])
    return pd.DataFrame()

def analyze_fa(df_quarter: pd.DataFrame):
    """FA filter bắt buộc (giống trước): EPS>500, ROE>10, 0<PE<10, Debt/Equity<1, CFO TTM +, LNST YoY +, tồn kho YoY <=30%."""
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
        inv   = f(latest, "inventory")
        liab  = f(latest, "liabilities")
        eq    = f(latest, "equity")

        lnst_q = pd.to_numeric(sub.get("netProfit"), errors="coerce").fillna(0.0).values.tolist()
        cfo_q  = pd.to_numeric(sub.get("netCashFlowFromOperatingActivities"), errors="coerce").fillna(0.0).values.tolist()

        inv_yoy = None
        if len(sub) >= 5:
            inv_yoy = f(sub.iloc[4].to_dict(), "inventory", None)

        # Điều kiện
        if not (0 < price < 10000): continue
        if not (eps > 500): continue
        if not (roe > 10):  continue
        if not (0 < pe < 10): continue
        if eq <= 0 or (liab/eq) >= 1.0: continue
        cfo_ttm = sum(cfo_q[:4]) if len(cfo_q) >= 4 else sum(cfo_q)
        if cfo_ttm <= 0: continue
        lnst_yoy_ok = False
        if len(lnst_q) >= 5:
            lnst_yoy_ok = lnst_q[0] > lnst_q[4]
        elif len(lnst_q) >= 8:
            lnst_yoy_ok = sum(lnst_q[:4]) > sum(lnst_q[4:8])
        if not lnst_yoy_ok: continue
        if inv_yoy and inv_yoy > 0:
            if (inv - inv_yoy) / inv_yoy > 0.30:
                continue

        fa_pass.append({
            "ticker": ticker,
            "price": price,
            "eps": eps,
            "roe": roe,
            "pe": pe
        })
    log(f"✅ FA PASS: {len(fa_pass)} mã")
    return fa_pass

# ============================================================
# B3) TA TỪ VNDIRECT (REALTIME MỖI LẦN CHẠY)
# ============================================================
def get_ohlc_days_vnd(ticker, days=120):
    start = (datetime.now() - timedelta(days=days*2)).strftime("%Y-%m-%d")
    params = {"q": f"ticker:{ticker}~date:gte:{start}", "sort": "date", "size": 1000}
    try:
        r = SESSION.get(PRICE_URL, params=params, timeout=(8, 18))
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        for c,n in [("adOpen","open"),("adHigh","high"),("adLow","low"),("adClose","close"),("adVolume","volume")]:
            if c in df.columns:
                df[n] = pd.to_numeric(df[c], errors="coerce")
        df = df[["date","open","high","low","close","volume"]].dropna().sort_values("date")
        return df.tail(150)
    except Exception as e:
        log(f"⚠️ OHLC {ticker} lỗi: {e}")
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

# ============================================================
# GỬI TELEGRAM
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

def format_msg(stocks):
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt FA + TA."
    msg = f"📈 [{today}] Mã <10k đạt FA + TA (≥3/5):\n\n"
    for s in stocks:
        msg += (f"• {s['ticker']} | Giá: {int(s['price'])}đ | EPS:{int(s['eps'])} "
                f"| ROE:{s['roe']:.1f}% | P/E:{s['pe']:.1f} | TA✓:{s['ta_score']}/5\n")
    return msg

# ============================================================
# MAIN MODES
#   - python main.py list   -> chỉ lấy danh sách <10k từ FireAnt
#   - python main.py fa     -> cập nhật & cache FA từ VNDIRECT
#   - python main.py scan   -> load FA cache -> quét TA realtime
# ============================================================
def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "scan").lower()
    log(f"🚀 Start BOT mode={mode}")

    if mode == "list":
        tks = get_tickers_under_10k_from_fireant()
        log(f"Done list: {len(tks)} mã")
        return

    if mode == "fa":
        tks = get_tickers_under_10k_from_fireant()
        if not tks:
            log("⚠️ Không có tickers từ FireAnt. Dừng FA update.")
            return
        _ = run_fa_update(tks)
        log("FA update DONE.")
        return

    # mode == scan (default): dùng FA cache + TA realtime
    df_fa_cache = load_fa_cache()
    if df_fa_cache.empty:
        log("🟡 FA cache trống → thử cập nhật nhanh")
        tks = get_tickers_under_10k_from_fireant()
        df_fa_cache = run_fa_update(tks) if tks else pd.DataFrame()

    fa_list = analyze_fa(df_fa_cache)
    if not fa_list:
        send_telegram("⚠️ BOT: Không có mã nào qua FA (cache rỗng hoặc dữ liệu thiếu).")
        return

    final = []
    for i, it in enumerate(fa_list, 1):
        tk = it["ticker"]
        log(f"[TA] {i}/{len(fa_list)} — {tk}")
        df = get_ohlc_days_vnd(tk, days=180)
        conds, score = technical_signals(df)
        if conds.get("enough_data") and score >= 3:
            final.append({**it, "ta_score": score})

    send_telegram(format_msg(final))
    log(f"ALL DONE. Final={len(final)}")

if __name__ == "__main__":
    main()
