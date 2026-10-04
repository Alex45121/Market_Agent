# Market Agent

A side project to predict short-term stock moves from price data and news, and then let an LLM agent use those predictions to make paper trades.

**Status:** the data pipeline and a first prediction model are built and tested. The LLM agent and paper trading are next.

## How it works

```
Alpaca API ──► hourly prices ──► DuckDB ──┐
                                          ├──► features ──► LightGBM model ──► prediction
Alpha Vantage API ──► news + sentiment ───┘                                        │
                         │                                                         ▼
                      FinBERT (open re-scoring)                     LLM agent (planned) ──► paper trades (Alpaca)
```

1. **Prices:** hourly bars for 15 large US stocks since 2016, from the Alpaca API, stored in DuckDB.
2. **News:** headlines with a per-stock sentiment score from Alpha Vantage. The free plan allows 25 requests a day and 1000 articles per request, so the downloader saves its progress and continues the next day, and splits any period that hits the article cap into halves.
3. **Features:** recent returns, moving-average ratios, RSI, volatility and volume, computed per stock.
4. **Model:** LightGBM predicting whether the price rises over the next 5 hours. The data is split by date, never randomly, so the model is always tested on a later period than it was trained on.
5. **Sentiment tests:** does news sentiment say anything about the next few hours? Tested on every stock with enough news, with control tests for news that only reacts to the price.

## Results so far

Details and charts are in [`notebooks/findings.ipynb`](notebooks/findings.ipynb).

| Test | Result |
|---|---|
| Price features only | AUC 0.54 (0.50 = coin flip) |
| + news sentiment as a feature | AUC 0.515 → 0.515, no improvement |
| Sentiment vs. next 1-24 h return | mean correlation ≈ 0 across 10 stocks |
| Sentiment vs. the return that already happened | 0.05-0.10, clearly higher |

**Conclusion:** the news mostly reacts to price moves that already happened, instead of predicting the next one. This held for Alpha Vantage's own score and for FinBERT. Many strongly scored headlines are also unrelated to short-term price moves, so better news filtering is the next thing to try.

## Project structure

```
src/
  config.py            stocks, paths, settings
  fetch_prices.py      Alpaca → DuckDB
  fetch_news.py        Alpha Vantage news, resumable, splits capped periods
  features.py          technical features and target
  train.py             time-based split, LightGBM, evaluation
  finbert_scores.py    re-scores headlines with FinBERT
  sentiment_tests.py   lag correlation and control tests
  plots.py             price vs sentiment charts, divergence search
notebooks/
  findings.ipynb       results and charts
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env      # add your Alpaca and Alpha Vantage keys

python -m src.fetch_prices
python -m src.fetch_news  # run once a day until all years are downloaded
python -m src.train
```

## Next steps

- Keep only news about real events (earnings, guidance, lawsuits, launches).
- Evaluate once on the held-out test period (from May 2024).
- Build the LLM agent: it reads the model's prediction plus current headlines, decides on paper trades through Alpaca, and logs its reasoning so every decision can be checked.

## Tools

Python, pandas, DuckDB, LightGBM, scikit-learn, SHAP, Hugging Face Transformers (FinBERT), Alpaca API, Alpha Vantage API.
