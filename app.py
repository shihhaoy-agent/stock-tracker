import io
import threading
import time

import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

_cache: dict = {}
_cache_lock = threading.Lock()

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def _read_html(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_cached(key: str, ttl: int, fetch_fn):
    with _cache_lock:
        if key in _cache:
            data, ts = _cache[key]
            if time.time() - ts < ttl:
                return data
    result = fetch_fn()
    with _cache_lock:
        _cache[key] = (result, time.time())
    return result


def fetch_sp500_tickers() -> list[tuple[str, str]]:
    tables = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    symbols = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    names = df["Security"].tolist()
    return list(zip(symbols, names))


def fetch_nasdaq100_tickers() -> list[tuple[str, str]]:
    tables = _read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for tbl in tables:
        cols = [c.lower() for c in tbl.columns]
        ticker_col = next((tbl.columns[i] for i, c in enumerate(cols) if c in ("ticker", "symbol")), None)
        name_col = next((tbl.columns[i] for i, c in enumerate(cols) if "company" in c or "name" in c or "security" in c), None)
        if ticker_col and name_col and len(tbl) > 50:
            symbols = tbl[ticker_col].str.replace(".", "-", regex=False).tolist()
            names = tbl[name_col].tolist()
            return list(zip(symbols, names))
    raise ValueError("Could not find NASDAQ-100 table on Wikipedia")


def fetch_stock_data(tickers_with_names: list[tuple[str, str]]) -> list[dict]:
    names = {t[0]: t[1] for t in tickers_with_names}
    results = []
    batch_size = 50  # keeps peak memory under 512MB on Render free tier

    for i in range(0, len(tickers_with_names), batch_size):
        batch = tickers_with_names[i : i + batch_size]
        tickers = [t[0] for t in batch]
        # Retry up to 3 times on rate limit
        for attempt in range(3):
            try:
                raw = yf.download(
                    tickers,
                    period="2d",
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                for ticker in tickers:
                    try:
                        if len(tickers) == 1:
                            close_s = raw["Close"].dropna()
                            vol_s = raw["Volume"].dropna()
                        else:
                            close_s = raw["Close"][ticker].dropna()
                            vol_s = raw["Volume"][ticker].dropna()

                        if len(close_s) < 2:
                            continue

                        prev = float(close_s.iloc[-2])
                        curr = float(close_s.iloc[-1])
                        pct = (curr - prev) / prev * 100
                        vol = int(vol_s.iloc[-1]) if len(vol_s) else 0

                        results.append({
                            "ticker": ticker,
                            "name": names.get(ticker, ticker),
                            "price": round(curr, 2),
                            "prev_close": round(prev, 2),
                            "change_abs": round(curr - prev, 2),
                            "change_pct": round(pct, 2),
                            "volume": vol,
                        })
                    except Exception:
                        continue
                del raw
                time.sleep(1)  # avoid rate limiting between batches
                break
            except Exception:
                time.sleep(3 * (attempt + 1))  # back off on failure

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stocks")
def stocks():
    idx = request.args.get("index", "sp500")

    if idx == "sp500":
        tickers = get_cached("sp500_tickers", 86_400, fetch_sp500_tickers)
    elif idx == "nasdaq100":
        tickers = get_cached("nasdaq100_tickers", 86_400, fetch_nasdaq100_tickers)
    else:
        sp = get_cached("sp500_tickers", 86_400, fetch_sp500_tickers)
        nq = get_cached("nasdaq100_tickers", 86_400, fetch_nasdaq100_tickers)
        seen = set()
        tickers = []
        for t in sp + nq:
            if t[0] not in seen:
                seen.add(t[0])
                tickers.append(t)

    data = get_cached(f"data_{idx}", 300, lambda: fetch_stock_data(tickers))

    return jsonify({
        "stocks": data,
        "count": len(data),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def _prewarm():
    """Fetch S&P 500 and NASDAQ-100 data in the background on startup."""
    time.sleep(2)  # let gunicorn finish booting
    for idx in ("sp500", "nasdaq100"):
        try:
            if idx == "sp500":
                tickers = get_cached("sp500_tickers", 86_400, fetch_sp500_tickers)
            else:
                tickers = get_cached("nasdaq100_tickers", 86_400, fetch_nasdaq100_tickers)
            get_cached(f"data_{idx}", 300, lambda t=tickers: fetch_stock_data(t))
        except Exception:
            pass


threading.Thread(target=_prewarm, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=False, port=5050)
