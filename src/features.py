"""Turn hourly price bars into model features and a prediction target."""
import pandas as pd
import ta

# Regular US trading session in UTC. 13-20 covers both winter (EST) and
# summer (EDT) time; pre- and after-market hours have very little volume.
MARKET_OPEN_UTC = 13
MARKET_CLOSE_UTC = 20

FORWARD_HOURS = 5
# Minimum 5-hour return that counts as "up" (target = 1).
# TODO: write down how this value was chosen.
UP_THRESHOLD = 0.0014246


def filter_market_hours(prices):
    hour = prices["timestamp"].dt.hour
    return prices[(hour >= MARKET_OPEN_UTC) & (hour <= MARKET_CLOSE_UTC)]


def add_technical_features(prices):
    """Per stock: recent returns, moving-average ratios, RSI, volatility, volume."""
    frames = []
    for symbol in prices["symbol"].unique():
        tdf = prices[prices["symbol"] == symbol].sort_values("timestamp").copy()

        tdf["return_1h"] = tdf["close"].pct_change()
        tdf["ma_10_ratio"] = tdf["close"] / tdf["close"].rolling(10).mean()
        tdf["ma_50_ratio"] = tdf["close"] / tdf["close"].rolling(50).mean()
        tdf["rsi"] = ta.momentum.RSIIndicator(tdf["close"], window=14).rsi()
        tdf["volatility_10"] = tdf["return_1h"].rolling(10).std()
        tdf["volume_ratio"] = tdf["volume"] / tdf["volume"].rolling(20).mean()
        for lag in [1, 2, 3]:
            tdf[f"return_lag_{lag}"] = tdf["return_1h"].shift(lag)

        frames.append(tdf)

    return pd.concat(frames, ignore_index=True).dropna()


def add_time_and_regime(data):
    data = data.copy()
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    # Volatility regime, only used to check the model in calm vs busy markets.
    data["vol_regime"] = pd.qcut(data["volatility_10"], 3, labels=["low", "mid", "high"])
    return data


def add_target(data, hours=FORWARD_HOURS, threshold=UP_THRESHOLD):
    """target = 1 if the price rises more than `threshold` over the next `hours` bars."""
    data = data.copy()
    data["next_return"] = data.groupby("symbol")["close"].transform(
        lambda close: close.pct_change(periods=hours).shift(-hours)
    )
    data["target"] = (data["next_return"] > threshold).astype(int)
    return data.dropna()


def build_dataset(prices):
    data = filter_market_hours(prices)
    data = add_technical_features(data)
    data = add_time_and_regime(data)
    return add_target(data)
