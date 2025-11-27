# ===========================
#  VN STOCK BOT: GOOGLE SHEET + VNDIRECT
#  - Google Sheet: danh sách mã cổ phiếu < 10.000đ (đã lọc sẵn)
#  - VNDIRECT: FA (financial_reports) + TA (stock_prices per ticker)
#  - Không dùng FireAnt nữa
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

TELEGRAM_TOKEN   = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()

# URL CSV của Google Sheet (đã lọc sẵn cp <10k)
# Ví dụ: https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0
SHEET_CSV_URL = (os.getenv("SHEET_CSV_URL") or "").strip()

CACHE_DIR = "/tmp/vnstock_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def log(msg: str):
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
# B1) DANH SÁCH MÃ TỪ GOOGLE SHEET (ĐÃ LỌC SẴN <10.000đ)
# ============================================================

def get_tickers_from_sheet():
    """
    Đọc Google Sheet CSV, lấy danh sách mã cổ phiếu.
    Sheet của bạn đã là cp <10.000đ nên KHÔNG lọc lại theo giá nữa.
    """
    url = SHEET_CSV_URL
    if not url:
        log("❌ SHEET_CSV_URL chưa cấu hình.")
        return []

    try:
        df = pd.read_csv(url, engine="python", on_bad_lines="skip")

        # Tìm cột 'Mã' hoặc các tên tương đương
        col = None
        for c in df.columns:
            if str(c).strip().lower() in ["mã", "ma", "ticker", "symbol", "code"]:
                col = c
                break
        if col is None:
            col = df.columns[0]  # fallback: cột đầu tiên

        tks = (df[col]
               .astype(str)
               .str.upper()
               .str.strip()
               .dropna()
               .unique()
               .tolist())
        tks = sorted(set(tks))
        log(f"✅ Sheet lấy được {len(tks)} mã cp <10k (đã lọc sẵn).")
        return tks
    except Exception as e:
        log(f"❌ Lỗi đọc sheet: {e}")
        return []

def get_tickers_under_10k(refresh: bool = False):
    """
    Hàm chuẩn để dùng trong main().
    Hiện tại: chỉ lấy từ Google Sheet, cache 30 phút.
    """
    cache_name = "tickers_from_sheet.json"
    if not refresh:
        cached = cache_get(cache_name, ttl_sec=1800)  # 30 phút
        if cached and cached.get("tickers"):
            log(f"🟢 Dùng cache tickers từ sheet: {len(cached['tickers'])} mã")
            return cached["tickers"]

    tks = get_tickers_from_sheet()
    cache_set(cache_name, {"tickers": tks})
    return tks

# ============================================================
# B2) FA TỪ VNDIRECT (CÓ CACHE 7 NGÀY)
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
    """
    FA filter:
      - 0 < price < 10000
      - EPS > 500
      - ROE > 10
      - 0 < PE < 10
      - Debt/Equity < 1
      - CFO TTM dương
      - LNST YoY tăng
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
# B3) TA: NẾN NGÀY TỪ VNDIRECT
# ============================================================

def get_ohlc_days_tcbs(ticker, days=180):
    tk = ticker.upper().strip()
    url = f"https://apipub.tcbs.com.vn/stock-insight/v1/stock/bars/{tk}"
    params = {"type":"stock","resolution":"1D","count":days}

    try:
        r = SESSION.get(url, params=params, timeout=(8,30))
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame([{
            "date": datetime.fromtimestamp(x["time"]/1000).date(),
            "open": x["open"],
            "high": x["high"],
            "low": x["low"],
            "close": x["close"],
            "volume": x["volume"]
        } for x in data])

        return df

    except Exception as e:
        log(f"⚠️ OHLC TCBS {tk} lỗi: {e}")
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

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    conds["ADX>20_DI+>DI-"]   = bool((latest["adx"] > 20) and (latest["di_pos"] > latest["di_neg"]))
    conds["RSI>50_cross_up"]  = bool((latest["rsi"] > 50) and (prev["rsi"] <= 50))
    conds["Break_20_high"]    = bool(latest["close"] > float(df["close"].iloc[-20:-1].max()))
    conds["Vol_up_3_days"]    = bool(df["volume"].iloc[-1] > df["volume"].iloc[-2] > df["volume"].iloc[-3])
    conds["Close>MA20_VolSp"] = bool(
        (latest["close"] > latest["ma20"]) and
        (latest["volume"] > 1.5 * latest["vol_ma20"])
    )

    score = sum(1 for v in conds.values() if v)
    conds["enough_data"] = True
    conds["score_TA_true"] = score
    return conds, score

# ============================================================
# GỬI TELEGRAM
# ============================================================
def send_telegram(text: str):
    token = TELEGRAM_TOKEN
    chat  = TELEGRAM_CHAT_ID
    if not token or not chat:
        log("❌ Thiếu TELEGRAM_TOKEN / TELEGRAM_CHAT_ID")
        return
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
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt FA + TA."
    msg = f"📈 [{today}] Mã <10k đạt FA + TA (≥3/5):\n\n"
    for s in stocks:
        msg += (
            f"• {s['ticker']} | Giá: {int(s['price'])}đ | EPS:{int(s['eps'])} "
            f"| ROE:{s['roe']:.1f}% | P/E:{s['pe']:.1f} | TA✓:{s['ta_score']}/5\n"
        )
    return msg

def format_msg_ta_only(stocks):
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt TA (≥3/5)."
    msg = f"📈 [{today}] Mã <10k đạt TA (≥3/5) – không lọc FA:\n\n"
    for s in stocks:
        msg += f"• {s['ticker']} | TA✓:{s['ta_score']}/5\n"
    return msg

# ============================================================
# MAIN MODES
#   - python main.py list   -> chỉ lấy danh sách mã từ Sheet
#   - python main.py fa     -> cập nhật & cache FA từ VNDIRECT
#   - python main.py scan   -> load FA cache -> quét TA + gửi Telegram
# ============================================================

def main():
    mode = (sys.argv[1] if len(sys.argv) > 1 else "scan").lower()
    log(f"🚀 Start BOT mode={mode}")

    # 1) Xem nhanh danh sách mã
    if mode == "list":
        tks = get_tickers_under_10k()
        log(f"Done list: {len(tks)} mã")
        return

    # 2) Cập nhật FA cache
    if mode == "fa":
        tks = get_tickers_under_10k()
        if not tks:
            log("⚠️ Không lấy được danh sách từ sheet.")
            return
        _ = run_fa_update(tks)
        log("FA update DONE.")
        return

    # 3) mode == scan (default): dùng FA cache + TA realtime
    df_fa_cache = load_fa_cache()
    fa_list = analyze_fa(df_fa_cache) if not df_fa_cache.empty else []

    if not fa_list:
        # 👉 TA-only: khi FA rỗng hoặc không pass
        log("🟠 Không dùng được FA → TA-only.")
        tks = get_tickers_under_10k()
        if not tks:
            send_telegram("⚠️ BOT: không lấy được danh sách mã từ sheet.")
            return
        final = []
        for i, tk in enumerate(tks, 1):
            log(f"[TA-only] {i}/{len(tks)} – {tk}")
            df = get_ohlc_days_vnd_per_ticker(tk, days=180)
            if df.empty:
                continue
            conds, score = technical_signals(df)
            if conds.get("enough_data") and score >= 3:
                final.append({"ticker": tk, "ta_score": score})
            time.sleep(0.15)
        send_telegram(format_msg_ta_only(final))
        log(f"ALL DONE (TA-only). Final={len(final)}")
        return

    # … nếu FA có dữ liệu thì chạy flow (FA -> TA)
    final = []
    for i, it in enumerate(fa_list, 1):
        tk = it["ticker"]
        log(f"[FA+TA] {i}/{len(fa_list)} — {tk}")
        df = get_ohlc_days_vnd_per_ticker(tk, days=180)
        if df.empty:
            continue
        conds, score = technical_signals(df)
        if conds.get("enough_data") and score >= 3:
            try:
                last_close = float(df["close"].iloc[-1])
            except Exception:
                last_close = it.get("price", 0)
            final.append({
                **it,
                "price": last_close,
                "ta_score": score
            })

    send_telegram(format_msg_fa_ta(final))
    log(f"ALL DONE. Final={len(final)}")

if __name__ == "__main__":
    main()
