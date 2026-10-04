"""Shared settings: which stocks to use and where the data lives."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "market_data_hourly.duckdb"
SENTIMENT_DIR = DATA_DIR / "sentiment"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # tech
    "META", "TSLA", "JPM", "V", "JNJ",        # tech, finance, healthcare
    "WMT", "PG", "XOM", "DIS", "KO",          # consumer, energy
]

# Ignore news articles that Alpha Vantage rates as only loosely about the stock.
MIN_RELEVANCE = 0.5
