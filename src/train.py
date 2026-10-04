"""Train a LightGBM model that predicts if a stock goes up over the next 5 hours.

Run from the repo root:  python -m src.train
"""
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score

from src.features import build_dataset
from src.fetch_prices import load_prices

FEATURES = [
    "return_1h", "ma_10_ratio", "ma_50_ratio", "rsi", "volatility_10", "volume_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3", "hour",
]


def time_split(data):
    """Split by date, never randomly, so the model is always tested on a later
    period than it was trained on. A random split would leak the future."""
    data = data.sort_values("timestamp")
    ts = data["timestamp"]
    train = data[(ts >= "2021-01-01") & (ts < "2023-09-01")]
    val = data[(ts >= "2023-09-01") & (ts < "2024-05-01")]
    test = data[ts >= "2024-05-01"]   # kept aside, not used yet
    return train, val, test


def train_model(train):
    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4, min_child_samples=50,
        class_weight="balanced", verbosity=-1,
    )
    model.fit(train[FEATURES], train["target"])
    return model


def evaluate(model, val):
    val_probs = model.predict_proba(val[FEATURES])[:, 1]
    val_preds = model.predict(val[FEATURES])

    print("AUC:", roc_auc_score(val["target"], val_probs))
    print(classification_report(val["target"], val_preds))

    # Does the model do better in calm or volatile markets?
    for regime in ["low", "mid", "high"]:
        mask = (val["vol_regime"] == regime).values
        print(regime, roc_auc_score(val["target"].values[mask], val_probs[mask]))

    return val_probs, val_preds


if __name__ == "__main__":
    data = build_dataset(load_prices())
    train, val, test = time_split(data)
    print("train:", len(train), "val:", len(val), "test:", len(test))
    model = train_model(train)
    evaluate(model, val)
