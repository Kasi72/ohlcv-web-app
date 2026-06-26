from http.server import BaseHTTPRequestHandler
import json
import io
import csv
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd
import yfinance as yf

# ── Aliases ──
NSE_ALIASES = {
    "NATCO": "NATCOPHARM", "NATACO": "NATCOPHARM",
    "HDFC": "HDFCBANK", "ICICI": "ICICIBANK", "KOTAK": "KOTAKBANK",
    "TATA": "TCS", "INFOSYS": "INFY",
    "TATA MOTORS": "TATAMOTORS", "MAHINDRA": "M_M", "M&M": "M_M",
    "HINDUSTAN UNILEVER": "HINDUNILVR",
    "LARSEN": "LT", "LARSEN & TOUBRO": "LT",
    "SBI": "SBIN", "BAJAJ FINANCE": "BAJFINANCE", "BAJAJ FINSERV": "BAJAJFINSV",
    "MARUTI": "MARUTI", "ADANI": "ADANIENT", "ADANI PORTS": "ADANIPORTS",
    "AXIS": "AXISBANK", "BHARTI": "BHARTIARTL", "AIRTEL": "BHARTIARTL",
    "DRREDDY": "DRREDDY", "CIPLA": "CIPLA", "HCLTECH": "HCLTECH",
    "TECHM": "TECHM", "HINDALCO": "HINDALCO", "JSWSTEEL": "JSWSTEEL",
    "APOLLOHOSP": "APOLLOHOSP", "TITAN": "TITAN", "WIPRO": "WIPRO",
}

def _lev(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]

def _norm(s: str) -> str:
    return "".join(ch for ch in s.upper().strip() if ch.isalnum())

def guess_alias(base: str, fuzzy: bool) -> str:
    base_n = _norm(base)
    if base in NSE_ALIASES: return NSE_ALIASES[base]
    if base_n in NSE_ALIASES: return NSE_ALIASES[base_n]
    for k, v in NSE_ALIASES.items():
        if _norm(k) == base_n: return v
    if not fuzzy: return base
    best_k, best_d = None, 999
    for k in NSE_ALIASES:
        d = _lev(base_n, _norm(k))
        if d < best_d: best_k, best_d = k, d
    if best_k and best_d <= 2:
        return NSE_ALIASES[best_k]
    return base

def resolve_symbol(raw: str, exchange: str, fuzzy: bool) -> str:
    s = raw.strip().upper()
    if not s: return ""
    for suf in (".NS", ".BO"):
        if s.endswith(suf):
            exchange = suf; s = s[:-3]; break
    mapped = guess_alias(s, fuzzy)
    if mapped.endswith((".NS", ".BO")): return mapped
    return f"{mapped}{exchange}"

def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1) if df.columns.nlevels > 1 else df.columns
    elif hasattr(df.columns, "to_flat_index"):
        flat = []
        for c in df.columns.to_flat_index():
            if isinstance(c, tuple):
                flat.append(c[0] if c and c[0] else "_".join(str(x) for x in c if x))
            else:
                flat.append(c)
        df.columns = flat
    return df

def normalize(df: pd.DataFrame, include_adj: bool) -> pd.DataFrame:
    df = flatten_cols(df).reset_index()
    # find date col
    date_col = None
    for name in ["date", "datetime", "timestamp", "Date"]:
        for c in df.columns:
            if str(c).strip().lower() == name.lower():
                date_col = str(c); break
        if date_col: break
    if not date_col and isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True); date_col = "index"
    if not date_col:
        date_col = str(df.columns[0])

    df = df.rename(columns={date_col: "DATE"})
    rmap = {"open": "OPEN", "high": "HIGH", "low": "LOW", "close": "CLOSE",
            "adj close": "ADJ_CLOSE", "adjclose": "ADJ_CLOSE", "volume": "VOLUME",
            "price": "CLOSE"}
    lm = {str(c).strip().lower(): c for c in df.columns}
    for k, v in rmap.items():
        if k in lm and v not in df.columns:
            df = df.rename(columns={lm[k]: v})
    req = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise RuntimeError(f"Missing: {missing}")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce", utc=False)
    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if include_adj and "ADJ_CLOSE" in df.columns:
        df["ADJ_CLOSE"] = pd.to_numeric(df["ADJ_CLOSE"], errors="coerce")
    df = df.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]).sort_values("DATE").reset_index(drop=True)
    cols = req[:]
    if include_adj and "ADJ_CLOSE" in df.columns: cols.append("ADJ_CLOSE")
    return df[cols]

def _fetch_yfinance(symbol: str, start: Optional[str], end: Optional[str],
                    auto_adjust: bool) -> pd.DataFrame:
    kwargs = {"interval": "1d", "progress": False, "auto_adjust": auto_adjust}
    if start and end:
        kwargs["start"] = start; kwargs["end"] = end
    else:
        kwargs["period"] = "max"
    df = yf.download(symbol, **kwargs)
    return df if df is not None else pd.DataFrame()

def _fetch_jugaad(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    from datetime import date as dt_date
    from jugaad_data.nse import stock_df
    bare = symbol.replace(".NS", "").replace(".BO", "")
    if start:
        sd = dt_date.fromisoformat(start)
    else:
        sd = dt_date(2000, 1, 1)
    if end:
        ed = dt_date.fromisoformat(end)
    else:
        ed = dt_date.today()
    df = stock_df(symbol=bare, from_date=sd, to_date=ed, series="EQ")
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def _build_result(df: pd.DataFrame, symbol: str, provider: str,
                  start: Optional[str], end: Optional[str],
                  include_adj: bool) -> Dict:
    df = normalize(df, include_adj)
    if start:
        df = df[df["DATE"] >= pd.to_datetime(start)]
    if end:
        df = df[df["DATE"] <= pd.to_datetime(end)]
    if df.empty:
        return {"ok": False, "symbol": symbol, "error": "Empty after date filter"}

    rows = len(df)
    d0, d1 = df["DATE"].min(), df["DATE"].max()
    bdays = pd.bdate_range(d0, d1).size

    records = []
    for _, row in df.iterrows():
        r = {
            "DATE": row["DATE"].strftime("%Y-%m-%d"),
            "OPEN": round(float(row["OPEN"]), 2),
            "HIGH": round(float(row["HIGH"]), 2),
            "LOW": round(float(row["LOW"]), 2),
            "CLOSE": round(float(row["CLOSE"]), 2),
            "VOLUME": int(row["VOLUME"]) if pd.notna(row["VOLUME"]) else 0,
        }
        if include_adj and "ADJ_CLOSE" in df.columns and pd.notna(row.get("ADJ_CLOSE")):
            r["ADJ_CLOSE"] = round(float(row["ADJ_CLOSE"]), 2)
        records.append(r)

    high_52 = float(df["HIGH"].tail(252).max()) if rows >= 10 else None
    low_52 = float(df["LOW"].tail(252).min()) if rows >= 10 else None

    return {
        "ok": True, "symbol": symbol, "provider": provider,
        "rows": rows,
        "start": d0.strftime("%Y-%m-%d"),
        "end": d1.strftime("%Y-%m-%d"),
        "missing_sessions": max(bdays - rows, 0),
        "last_close": round(float(df["CLOSE"].iloc[-1]), 2),
        "avg_volume": round(float(df["VOLUME"].mean()), 0),
        "high_52w": round(high_52, 2) if high_52 else None,
        "low_52w": round(low_52, 2) if low_52 else None,
        "data": records,
    }

def fetch_symbol(symbol: str, start: Optional[str], end: Optional[str],
                 auto_adjust: bool, include_adj: bool) -> Dict:
    errors = []

    # Try yfinance first
    try:
        df = _fetch_yfinance(symbol, start, end, auto_adjust)
        if df is not None and not df.empty:
            return _build_result(df, symbol, "yfinance", start, end, include_adj)
        else:
            errors.append("yfinance: empty result")
    except Exception as e:
        errors.append(f"yfinance: {e}")

    # Fallback to jugaad-data (NSE direct)
    try:
        df = _fetch_jugaad(symbol, start, end)
        if df is not None and not df.empty:
            return _build_result(df, symbol, "jugaad-data (NSE)", start, end, include_adj)
        else:
            errors.append("jugaad-data: empty result")
    except Exception as e:
        errors.append(f"jugaad-data: {e}")

    return {"ok": False, "symbol": symbol, "error": " | ".join(errors)}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        raw_symbols = body.get("symbols", [])
        exchange = body.get("exchange", ".NS")
        fuzzy = body.get("fuzzy", True)
        start = body.get("start")
        end = body.get("end")
        auto_adjust = body.get("auto_adjust", False)
        include_adj = body.get("include_adj", False)

        seen, symbols = set(), []
        for r in raw_symbols:
            sym = resolve_symbol(r, exchange, fuzzy)
            if sym and sym not in seen:
                symbols.append(sym); seen.add(sym)

        results = []
        for sym in symbols[:20]:
            results.append(fetch_symbol(sym, start, end, auto_adjust, include_adj))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"results": results}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
