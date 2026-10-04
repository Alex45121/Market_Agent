"""Download news sentiment from Alpha Vantage for each stock, 2016-2025.

The free plan has two limits that shape this script:
- 25 requests per day, so the script saves its progress and continues where
  it stopped when you run it again the next day.
- At most 1000 articles per request. If a period hits that cap, it is split
  in half and both halves are requested separately.

Run from the repo root:  python -m src.fetch_news
"""
import glob
import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import SENTIMENT_DIR, TICKERS

URL = "https://www.alphavantage.co/query"
ARTICLE_CAP = 1000
MAX_CALLS_PER_RUN = 16      # stays safely under the 25-per-day limit
SECONDS_BETWEEN_CALLS = 15
MIN_SPLIT_DAYS = 31         # stop splitting below about a month
YEARS = range(2016, 2026)
PROGRESS_FILE = SENTIMENT_DIR / "progress.json"


class RateLimited(Exception):
    pass


def request_news(ticker, start, end, api_key):
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "time_from": start.strftime("%Y%m%dT%H%M"),
        "time_to": end.strftime("%Y%m%dT%H%M"),
        "limit": ARTICLE_CAP,
        "apikey": api_key,
    }
    data = requests.get(URL, params=params, timeout=30).json()
    # When the limit is hit, Alpha Vantage answers with a message instead of
    # articles. The message contains the API key, so it is not printed.
    if "Information" in data or "Note" in data:
        raise RateLimited()
    return data.get("feed", [])


def extract_ticker_sentiment(articles, ticker):
    """Keep only the sentiment score for the stock we asked about.

    One article can mention several stocks, each with its own score.
    """
    rows = []
    for article in articles:
        for ts in article.get("ticker_sentiment", []):
            if ts["ticker"] == ticker:
                rows.append({
                    "ticker": ticker,
                    "time_published": article["time_published"],
                    "title": article["title"],
                    "relevance": float(ts["relevance_score"]),
                    "ticker_sentiment_score": float(ts["ticker_sentiment_score"]),
                    "ticker_sentiment_label": ts["ticker_sentiment_label"],
                })
    return pd.DataFrame(rows)


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=1))


def period_key(ticker, start, end):
    return f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}"


def split_in_half(ticker, start, end):
    mid = (start + (end - start) / 2).floor("D")
    return [(ticker, start, mid), (ticker, mid + pd.Timedelta(minutes=1), end)]


def fetch_all(api_key):
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
    progress = load_progress()   # key -> "saved" or "split"

    todo = [
        (t, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31 23:59"))
        for t in TICKERS for y in YEARS
    ]
    calls = 0

    while todo:
        ticker, start, end = todo.pop(0)
        key = period_key(ticker, start, end)

        if progress.get(key) == "saved":
            continue
        if progress.get(key) == "split":
            # Already known to be too big: go straight to the halves.
            todo[:0] = split_in_half(ticker, start, end)
            continue
        if calls >= MAX_CALLS_PER_RUN:
            print("Reached the call limit for this run. Run again tomorrow.")
            break

        try:
            articles = request_news(ticker, start, end, api_key)
        except RateLimited:
            print("Alpha Vantage rate limit reached. Run again tomorrow.")
            break
        calls += 1

        capped = len(articles) >= ARTICLE_CAP
        if capped and (end - start).days > MIN_SPLIT_DAYS:
            print(f"{key}: hit the {ARTICLE_CAP}-article cap, splitting in half")
            progress[key] = "split"
            todo[:0] = split_in_half(ticker, start, end)
        else:
            df = extract_ticker_sentiment(articles, ticker)
            if not df.empty:
                df.to_csv(SENTIMENT_DIR / f"{key}.csv", index=False)
            note = " (still capped)" if capped else ""
            print(f"{key}: {len(articles)} articles{note} ({calls}/{MAX_CALLS_PER_RUN})")
            progress[key] = "saved"

        save_progress(progress)
        time.sleep(SECONDS_BETWEEN_CALLS)

    remaining = sum(1 for t in TICKERS for y in YEARS
                    if progress.get(period_key(t, pd.Timestamp(f"{y}-01-01"),
                                               pd.Timestamp(f"{y}-12-31 23:59"))) is None)
    print(f"Years not started yet: {remaining} of {len(TICKERS) * len(YEARS)}")


def load_news(sentiment_dir=SENTIMENT_DIR):
    """Load every saved CSV into one table, with time_published as a datetime."""
    files = glob.glob(os.path.join(sentiment_dir, "*.csv"))
    news = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    news["time_published"] = pd.to_datetime(news["time_published"], format="%Y%m%dT%H%M%S")
    return news


if __name__ == "__main__":
    load_dotenv()
    fetch_all(os.getenv("ALPHA_VANTAGE_KEY"))
