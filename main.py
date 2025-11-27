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
from vnstock import Vnstock  # pip install vnstock

# ---------- ENV & CACHE DIR ----------
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

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
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [{today}] Không có mã nào đạt FA (vnstock) + TA (≥3/5)."
    msg = f"📈 [{today}] Mã <10k đạt FA (vnstock) + TA (≥3/5):\n\n"
    for s in stocks:
        de_txt = f"{s['de']:.2f}" if s.get("de") is not None else "N/A"
        msg += (
            f"• {s['ticker']} | EPS:{int(s['eps'])} | ROE:{s['roe']:.1f}% "
            f"| P/E:{s['pe']:.1f} | D/E:{de_txt} | TA✓:{s['ta_score']}/5\n"
        )
    return msg

def format_msg_ta_only(stocks):
    today = datetime.now().strftime("%d/%m/%Y")
    if not stocks:
        return f"📉 [{today}] Mã CP <30k đạt TA (≥3/5) – không lọc FA."
    msg = f"📊 [{today}] Mã CP <30k đạt TA (≥3/5) – không lọc FA:\n\n"
    for s in stocks:
        msg += f"• {s['ticker']} | TA✓:{s['ta_score']}/5\n"
    return msg

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
    # ==== CHỈ UPDATE FA LÚC 19H THỨ 6 (GIỜ VIỆT NAM) ====
    # Giả sử cron của bạn chạy MỖI NGÀY lúc 19h VN (12h UTC)
    now_utc = datetime.utcnow()
    now_vn  = now_utc + timedelta(hours=7)   # đổi sang giờ VN

    if now_vn.weekday() == 4 and now_vn.hour == 19:   # 4 = Friday, 19h
        log("🔄 Thứ 6 19h VN → CẬP NHẬT FA (vnstock)…")
        run_fa_update_vnstock(tks)   # chỉ chạy ở thời điểm này
    else:
        log("⏭ Không phải 19h Thứ 6 → dùng FA cache cũ, không update.")
    # ==== MODE FA: chỉ cập nhật FA rồi dừng ====
    if mode == "fa":
        log("🔄 Cập nhật FA (vnstock) cho danh sách mã …")
        run_fa_update_vnstock(tks)     # <-- CẬP NHẬT FA & LƯU CACHE
        log("FA update DONE.")
        return

    # --- mode: scan (FA + TA nếu có FA, else TA-only) ---
    # log("🔄 Cập nhật FA (vnstock) trước khi scan TA…")
    # run_fa_update_vnstock(tks)
    
    df_fa_cache = load_fa_cache()
    fa_list = analyze_fa(df_fa_cache) if not df_fa_cache.empty else []

    # --- mode: scan (FA + TA nếu có FA, else TA-only) ---
    if mode == "scan":
        log("🔄 Cập nhật FA (vnstock) trước khi scan TA…")
        run_fa_update_vnstock(tks)   # ✅ luôn update FA trước
    
        df_fa_cache = load_fa_cache()
        fa_list = analyze_fa(df_fa_cache) if not df_fa_cache.empty else []

    if not fa_list:
        log("🟠 Không dùng được FA (cache rỗng hoặc không mã nào pass) → TA-only.")
        final = []
        for i, tk in enumerate(tks, 1):
            log(f"[TA-only] {i}/{len(tks)} – {tk}")
            df = get_ohlc_days_tcbs(tk, days=180)
            if df.empty:
                continue
            conds, score = technical_signals(df)
            if conds.get("enough_data") and score >= 3:
                final.append({"ticker": tk, "ta_score": score})
            time.sleep(0.15)
        send_telegram(format_msg_ta_only(final))
        log(f"ALL DONE (TA-only). Final={len(final)}")
        return

    # Nếu FA có dữ liệu thì chạy FA -> TA
    final = []
    for i, it in enumerate(fa_list, 1):
        tk = it["ticker"]
        log(f"[FA+TA] {i}/{len(fa_list)} — {tk}")
        df = get_ohlc_days_tcbs(tk, days=180)
        if df.empty:
            continue
        conds, score = technical_signals(df)
        if conds.get("enough_data") and score >= 3:
            final.append({**it, "ta_score": score})
        time.sleep(0.15)

    send_telegram(format_msg_fa_ta(final))
    log(f"ALL DONE (FA+TA). Final={len(final)}")

if __name__ == "__main__":
    main()
