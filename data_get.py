import ccxt
import pandas as pd

def get_data():
    exchange = ccxt.binance({
        "enableRateLimit": True
    })

    def fetch_all_ohlcv(symbol, timeframe, since, limit=1000):
        all_candles = []

        while True:
            candles = exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            if not candles:
                break

            all_candles.extend(candles)

            since = candles[-1][0] + 1

            print(f"Загружено свечей: {len(all_candles)}")

            if len(candles) < limit:
                break

        return all_candles


    since = exchange.parse8601("2025-01-01T00:00:00Z")

    candles = fetch_all_ohlcv(
        "BTC/USDT",
        "1h",
        since
    )

    cols = ["date", "open", "high", "low", "close", "volume"]

    data_raw = pd.DataFrame(candles, columns=cols)

    data_raw["date"] = pd.to_datetime(data_raw["date"], unit="ms", utc=True)

    data_raw = data_raw.drop_duplicates(subset="date")
    data_raw = data_raw.sort_values("date")

    data = data_raw.set_index("date")
    data = data.iloc[:-1]

    print(data.shape)
    print(data.head())
    print(data.tail())
    return data