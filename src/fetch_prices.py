"""Download hourly price bars from Alpaca and store them in DuckDB.

Run from the repo root:  python -m src.fetch_prices
"""
import os

import duckdb
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from src.config import DATA_DIR, DB_PATH, TICKERS


def fetch_hourly_bars(tickers, start="2016-01-01"):
    load_dotenv()
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY"),
    )
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Hour,
        start=start,
        adjustment="all",  # adjust for splits and dividends
    )
    return client.get_stock_bars(request).df.reset_index()


def save_prices(bars, db_path=DB_PATH):
    DATA_DIR.mkdir(exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE OR REPLACE TABLE price_bars AS SELECT * FROM bars")
    con.close()


def load_prices(db_path=DB_PATH):
    """Load all bars, with timestamps in UTC (timezone removed)."""
    con = duckdb.connect(str(db_path), read_only=True)
    prices = con.execute("SELECT * FROM price_bars").df()
    con.close()
    # DuckDB returns timestamps in the local timezone, so convert to UTC
    # before dropping the timezone. Otherwise the hours shift by 1-2 h.
    prices["timestamp"] = prices["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return prices


if __name__ == "__main__":
    bars = fetch_hourly_bars(TICKERS)
    print(f"Downloaded {len(bars):,} bars for {bars['symbol'].nunique()} stocks")
    save_prices(bars)
    print(f"Saved to {DB_PATH}")
