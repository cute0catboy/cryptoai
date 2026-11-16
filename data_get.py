import ccxt
import pandas as pd

def get_data():
    exchange = ccxt.binance()
    data_raw0 = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=1000)

    cols = ["date", "open", "high", "low", "close", "volume"]
    data_raw = pd.DataFrame(data=data_raw0, columns=cols)

    dt_series = pd.to_datetime(data_raw["date"], unit="ms")

    data = pd.DataFrame(
        data=data_raw.loc[:, "open":].to_numpy(),
        index=dt_series,
        columns=cols[1:],
    )
    return data