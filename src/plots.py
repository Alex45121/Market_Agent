"""Charts for comparing price moves with news sentiment, plus a search for
cases where they clearly disagree."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MIN_RELEVANCE
from src.sentiment_tests import weighted_sentiment

FORWARD_HORIZON = 5
VOL_WINDOW = 50
VOL_FLOOR_PCTILE = 0.10
EPS = 1e-6


def plot_price_vs_cumulative_sentiment(ticker, start_date, end_date, prices, news):
    """Price next to the running total of weighted sentiment."""
    p = prices[(prices["symbol"] == ticker) & (prices["timestamp"] >= start_date)
               & (prices["timestamp"] <= end_date)].sort_values("timestamp")
    n = news[(news["ticker"] == ticker) & (news["time_published"] >= start_date)
             & (news["time_published"] <= end_date)
             & (news["relevance"] > MIN_RELEVANCE)].sort_values("time_published").copy()
    if n.empty:
        print(f"No news for {ticker} in this range.")
        return

    n["ticker_sentiment_score"] = pd.to_numeric(n["ticker_sentiment_score"], errors="coerce")
    n["cumulative_sentiment"] = n["ticker_sentiment_score"].apply(weighted_sentiment).cumsum()
    print(f"{ticker}: {(n['ticker_sentiment_score'] > 0).sum()} positive, "
          f"{(n['ticker_sentiment_score'] < 0).sum()} negative articles (relevance > {MIN_RELEVANCE})")

    fig, ax1 = plt.subplots(figsize=(16, 7))
    ax1.plot(p["timestamp"], p["close"], color="black", linewidth=1)
    ax1.set_ylabel("Close price")
    ax2 = ax1.twinx()
    ax2.plot(n["time_published"], n["cumulative_sentiment"], color="steelblue", linewidth=1.2)
    ax2.set_ylabel("Cumulative weighted sentiment", color="steelblue")
    plt.title(f"{ticker}: price vs cumulative weighted sentiment")
    fig.tight_layout()
    plt.show()


def price_move_line(ticker, prices, start_date, end_date):
    """Next-5-hour return divided by the stock's recent volatility, squashed to -1..1.

    So a value of 0.9 means "a big move for this stock right now", whatever its price level.
    """
    p = prices[prices["symbol"] == ticker].sort_values("timestamp").copy()
    p["return_1h"] = p["close"].pct_change()
    vol = p["return_1h"].rolling(VOL_WINDOW).std()
    vol = vol.clip(lower=vol.quantile(VOL_FLOOR_PCTILE))  # avoid dividing by ~0 in quiet periods
    fwd = p["close"].pct_change(periods=FORWARD_HORIZON).shift(-FORWARD_HORIZON)
    p["move_norm"] = np.tanh(fwd / (vol * np.sqrt(FORWARD_HORIZON) + EPS))
    p = p[(p["timestamp"] >= start_date) & (p["timestamp"] <= end_date)]
    return p[["timestamp", "move_norm"]].dropna()


def sentiment_line(ticker, news, start_date, end_date):
    """Mean sentiment per hour (mean, not sum, so it stays between -1 and 1)."""
    n = news[(news["ticker"] == ticker) & (news["relevance"] > MIN_RELEVANCE)
             & (news["time_published"] >= start_date)
             & (news["time_published"] <= end_date)].copy()
    if n.empty:
        return pd.DataFrame(columns=["timestamp", "sentiment_mean"])
    n["ticker_sentiment_score"] = pd.to_numeric(n["ticker_sentiment_score"], errors="coerce")
    n["timestamp"] = n["time_published"].dt.floor("h")
    return (n.groupby("timestamp")["ticker_sentiment_score"].mean()
             .reset_index(name="sentiment_mean"))


def plot_dual_normalized(ticker, start_date, end_date, news, prices, label="AV sentiment",
                         price_smooth=6, sentiment_smooth=3):
    """Price move and sentiment on the same -1..1 scale.

    The smoothing uses a centred window, which looks slightly into the future.
    That is fine for a chart but must never be used as a model input.
    """
    price = price_move_line(ticker, prices, start_date, end_date).set_index("timestamp")
    sent = sentiment_line(ticker, news, start_date, end_date)
    if price.empty and sent.empty:
        print(f"No data for {ticker} in this range.")
        return

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(price.index, price["move_norm"], color="black", linewidth=0.4, alpha=0.25)
    ax.plot(price.index, price["move_norm"].rolling(price_smooth, center=True, min_periods=1).mean(),
            color="black", linewidth=1.5, label=f"normalized price move ({price_smooth}h smoothed)")
    if not sent.empty:
        # Smooth over consecutive news hours, not calendar hours (news is sparse).
        sent = sent.sort_values("timestamp").set_index("timestamp")
        ax.plot(sent.index,
                sent["sentiment_mean"].rolling(sentiment_smooth, center=True, min_periods=1).mean(),
                color="steelblue", linewidth=1.8, marker="o", markersize=3,
                label=f"{label} ({sentiment_smooth}-article smoothed)")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("-1 to 1 (both series)")
    ax.legend(loc="upper left")
    plt.title(f"{ticker}: normalized price move vs {label}")
    fig.tight_layout()
    plt.show()


PRICE_MOVE_THRESHOLD = 0.7   # |move_norm| above this is a "big" move
SENTIMENT_THRESHOLD = 0.2    # |sentiment| above this is clearly positive or negative
HEADLINE_WINDOW_HOURS = 6


def find_divergences(ticker, news, prices, start_date, end_date, top_n=10):
    """Big price moves where the news around them pointed the other way.

    Printing the actual headlines shows whether the news was wrong or whether
    the sentiment score misread it.
    """
    price = price_move_line(ticker, prices, start_date, end_date)
    if price.empty:
        print(f"No price data for {ticker} in this range.")
        return pd.DataFrame()

    window_h = pd.Timedelta(hours=HEADLINE_WINDOW_HOURS)
    n = news[(news["ticker"] == ticker) & (news["relevance"] > MIN_RELEVANCE)
             & (news["time_published"] >= pd.Timestamp(start_date) - window_h)
             & (news["time_published"] <= pd.Timestamp(end_date) + window_h)].copy()
    n["ticker_sentiment_score"] = pd.to_numeric(n["ticker_sentiment_score"], errors="coerce")

    rows = []
    for _, move in price[price["move_norm"].abs() > PRICE_MOVE_THRESHOLD].iterrows():
        t = move["timestamp"]
        nearby = n[(n["time_published"] >= t - window_h) & (n["time_published"] <= t + window_h)]
        if nearby.empty:
            continue
        avg = nearby["ticker_sentiment_score"].mean()
        up_but_negative = move["move_norm"] > PRICE_MOVE_THRESHOLD and avg < -SENTIMENT_THRESHOLD
        down_but_positive = move["move_norm"] < -PRICE_MOVE_THRESHOLD and avg > SENTIMENT_THRESHOLD
        if not (up_but_negative or down_but_positive):
            continue

        strongest = nearby.sort_values("ticker_sentiment_score", key=abs, ascending=False).iloc[0]
        rows.append({
            "timestamp": t,
            "direction": "PRICE UP / sentiment NEGATIVE" if up_but_negative
                         else "PRICE DOWN / sentiment POSITIVE",
            "move_norm": move["move_norm"],
            "avg_sentiment": avg,
            "n_articles": len(nearby),
            "top_headline": strongest["title"],
        })

    results = pd.DataFrame(rows)
    if results.empty:
        print(f"{ticker}: no strong divergences in this window.")
        return results

    results = results.sort_values("avg_sentiment", key=abs, ascending=False).head(top_n)
    pd.set_option("display.max_colwidth", 100)
    print(f"\n=== {ticker}: top {len(results)} divergence cases ===")
    print(results.to_string(index=False))
    return results
