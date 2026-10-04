"""Does news sentiment say anything about where the price goes next?

Two tests, run for every stock with enough news (no picking the ones that look good):

1. Lag correlation: hourly sentiment vs. the return over the next 1-24 hours.
2. Control tests, to check whether a correlation is real:
   - backward: sentiment vs. the return that ALREADY happened. If this is as
     strong as the forward one, the news is just reacting to the price move.
   - article count: number of articles vs. forward return. If this is as strong,
     the sentiment score adds nothing beyond "there was news".
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MIN_RELEVANCE

HORIZONS = [1, 2, 4, 8, 24]

# Weights for positive and negative news, chosen earlier by looking at charts
# for a few stocks. Kept fixed here (not tuned per stock) so the test is fair.
POS_MULT = 0.24252627
NEG_MULT = 1.6118 ** (-POS_MULT ** 2)


def weighted_sentiment(score):
    if pd.isna(score):
        return 0.0
    score = float(score)
    return score * NEG_MULT if score < 0 else score * POS_MULT


def relevant_news(ticker, news):
    sub = news[(news["ticker"] == ticker) & (news["relevance"] > MIN_RELEVANCE)].copy()
    sub["hour_bucket"] = sub["time_published"].dt.floor("h")
    return sub


def hourly_sentiment(ticker, news):
    """One weighted sentiment value per hour (articles in the same hour are summed)."""
    sub = relevant_news(ticker, news)
    if sub.empty:
        return None
    sub["ticker_sentiment_score"] = pd.to_numeric(sub["ticker_sentiment_score"], errors="coerce")
    sub["weighted_score"] = sub["ticker_sentiment_score"].apply(weighted_sentiment)
    hourly = sub.groupby("hour_bucket")["weighted_score"].sum().reset_index()
    return hourly.rename(columns={"hour_bucket": "timestamp"})


def hourly_article_count(ticker, news):
    sub = relevant_news(ticker, news)
    if sub.empty:
        return None
    hourly = sub.groupby("hour_bucket").size().reset_index(name="article_count")
    return hourly.rename(columns={"hour_bucket": "timestamp"})


def forward_returns(ticker, prices, horizons=HORIZONS):
    p = prices[prices["symbol"] == ticker].sort_values("timestamp").set_index("timestamp")
    out = pd.DataFrame(index=p.index)
    for h in horizons:
        out[f"fwd_return_{h}h"] = p["close"].shift(-h) / p["close"] - 1
    return out.reset_index()


def backward_returns(ticker, prices, horizons=HORIZONS):
    p = prices[prices["symbol"] == ticker].sort_values("timestamp").set_index("timestamp")
    out = pd.DataFrame(index=p.index)
    for h in horizons:
        out[f"bwd_return_{h}h"] = p["close"] / p["close"].shift(h) - 1
    return out.reset_index()


def _prepare(df):
    # merge_asof needs the same datetime resolution on both sides
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")
    return df.sort_values("timestamp")


def run_lag_correlation_test(tickers, news, prices, horizons=HORIZONS):
    results = []
    for ticker in tickers:
        sent = hourly_sentiment(ticker, news)
        if sent is None or len(sent) < 10:
            print(f"{ticker}: not enough news, skipped")
            continue

        # Match each news hour to the next price bar, at most 6 hours later.
        merged = pd.merge_asof(_prepare(sent), _prepare(forward_returns(ticker, prices, horizons)),
                               on="timestamp", direction="forward",
                               tolerance=pd.Timedelta(hours=6))
        merged = merged.dropna(subset=["weighted_score"])

        row = {"ticker": ticker, "n_events": len(merged)}
        for h in horizons:
            valid = merged.dropna(subset=[f"fwd_return_{h}h"])
            row[f"corr_{h}h"] = (valid["weighted_score"].corr(valid[f"fwd_return_{h}h"])
                                 if len(valid) >= 10 else np.nan)
            row[f"n_{h}h"] = len(valid)
        results.append(row)

    return pd.DataFrame(results)


def run_control_tests(tickers, news, prices, horizons=HORIZONS):
    rows = []
    for ticker in tickers:
        sent = hourly_sentiment(ticker, news)
        counts = hourly_article_count(ticker, news)
        if sent is None or len(sent) < 10:
            print(f"{ticker}: not enough news, skipped")
            continue

        m = pd.merge_asof(_prepare(sent), _prepare(counts), on="timestamp",
                          direction="nearest", tolerance=pd.Timedelta(minutes=30))
        m = pd.merge_asof(m, _prepare(forward_returns(ticker, prices, horizons)), on="timestamp",
                          direction="forward", tolerance=pd.Timedelta(hours=6))
        m = pd.merge_asof(m, _prepare(backward_returns(ticker, prices, horizons)), on="timestamp",
                          direction="forward", tolerance=pd.Timedelta(hours=6))
        m = m.dropna(subset=["weighted_score"])

        for h in horizons:
            fcol, bcol = f"fwd_return_{h}h", f"bwd_return_{h}h"
            vf, vb = m.dropna(subset=[fcol]), m.dropna(subset=[bcol])
            rows.append({
                "ticker": ticker,
                "horizon_h": h,
                "corr_fwd": vf["weighted_score"].corr(vf[fcol]) if len(vf) >= 10 else np.nan,
                "corr_bwd": vb["weighted_score"].corr(vb[bcol]) if len(vb) >= 10 else np.nan,
                "corr_cnt": vf["article_count"].corr(vf[fcol]) if len(vf) >= 10 else np.nan,
            })

    return pd.DataFrame(rows)


def print_summary(results, horizons=HORIZONS):
    print("\n--- Aggregate across tickers (mean, std, sign consistency) ---")
    for h in horizons:
        vals = results[f"corr_{h}h"].dropna()
        if len(vals) == 0:
            continue
        print(f"{h}h: mean corr = {vals.mean():.4f}, std = {vals.std():.4f}, "
              f"n_tickers = {len(vals)}, positive in {(vals > 0).sum()}/{len(vals)}, "
              f"negative in {(vals < 0).sum()}/{len(vals)}")


def print_control_summary(control, horizons=HORIZONS):
    print("\n--- Aggregate by horizon ---")
    for h in horizons:
        sub = control[control["horizon_h"] == h]
        n = sub["corr_fwd"].notna().sum()
        print(f"{h}h: mean corr_fwd={sub['corr_fwd'].mean():.4f}  "
              f"corr_bwd={sub['corr_bwd'].mean():.4f}  corr_cnt={sub['corr_cnt'].mean():.4f}   "
              f"(fwd>bwd: {(sub['corr_fwd'] > sub['corr_bwd']).sum()}/{n},  "
              f"fwd>cnt: {(sub['corr_fwd'] > sub['corr_cnt']).sum()}/{n})")


def plot_correlation_bars(results, horizons=HORIZONS):
    df = results.dropna(subset=[f"corr_{h}h" for h in horizons], how="all")
    if df.empty:
        print("No stocks with enough data to plot.")
        return

    x = np.arange(len(df))
    width = 0.8 / len(horizons)
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.2), 6))
    for i, h in enumerate(horizons):
        ax.bar(x + i * width, df[f"corr_{h}h"].values, width, label=f"{h}h")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(horizons) - 1) / 2)
    ax.set_xticklabels(df["ticker"], rotation=45, ha="right")
    ax.set_ylabel("Correlation (weighted sentiment vs forward return)")
    ax.set_title("Sentiment-weight → forward-return correlation, by ticker and horizon")
    ax.legend(title="horizon")
    fig.tight_layout()
    plt.show()
