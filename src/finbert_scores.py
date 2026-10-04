"""Score news headlines with FinBERT as an open alternative to Alpha Vantage's score.

Alpha Vantage does not explain how its sentiment score is made. FinBERT is an
open model, so running it on the same headlines shows whether the results
depend on the scoring method.

score = P(positive) - P(negative), roughly -1 to 1, same scale as Alpha Vantage.
Scores are cached in a CSV so headlines are only scored once.
"""
import os

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DATA_DIR

MODEL_NAME = "ProsusAI/finbert"
CACHE_FILE = DATA_DIR / "finbert_title_scores.csv"
BATCH_SIZE = 64


def load_finbert():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


def score_titles(titles, tokenizer, model, device, batch_size=BATCH_SIZE):
    # ProsusAI/finbert label order: 0 = positive, 1 = negative, 2 = neutral
    results = []
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            batch = titles[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=64,
                               return_tensors="pt").to(device)
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()

            for title, (pos, neg, neu) in zip(batch, probs):
                results.append({
                    "title": title,
                    "finbert_pos": float(pos),
                    "finbert_neg": float(neg),
                    "finbert_neutral": float(neu),
                    "finbert_score": float(pos - neg),
                })
            if (i // batch_size) % 20 == 0:
                print(f"  {i + len(batch)}/{len(titles)} titles scored")

    return pd.DataFrame(results)


def build_news_finbert(news):
    """Return a copy of `news` where ticker_sentiment_score is the FinBERT score,
    so every analysis function can be run on it without changes."""
    titles = news["title"].dropna().unique().tolist()

    if os.path.exists(CACHE_FILE):
        cached = pd.read_csv(CACHE_FILE)
        remaining = [t for t in titles if t not in set(cached["title"])]
    else:
        cached = pd.DataFrame(columns=["title", "finbert_pos", "finbert_neg",
                                       "finbert_neutral", "finbert_score"])
        remaining = titles
    print(f"{len(titles)} unique titles, {len(remaining)} not scored yet")

    if remaining:
        tokenizer, model, device = load_finbert()
        scores = pd.concat([cached, score_titles(remaining, tokenizer, model, device)],
                           ignore_index=True)
        scores.to_csv(CACHE_FILE, index=False)
    else:
        scores = cached

    news_finbert = news.merge(
        scores[["title", "finbert_score", "finbert_pos", "finbert_neg", "finbert_neutral"]],
        on="title", how="left",
    ).dropna(subset=["finbert_score"])
    news_finbert["ticker_sentiment_score"] = news_finbert["finbert_score"]
    return news_finbert
