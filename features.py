import numpy as np


def get_rsi(df, period=15):
    delta = df["c"].diff()
    up_days = delta.copy()
    up_days[delta <= 0] = 0.0
    down_days = abs(delta.copy())
    down_days[delta > 0] = 0.0
    RS_up = up_days.rolling(period).mean()
    RS_down = down_days.rolling(period).mean()
    rsi = 100 - 100 / (1 + RS_up / RS_down)

    return rsi


def get_stoch_rsi(series, period=14):
    # Calculate RSI
    delta = series.diff().dropna()
    ups = delta * 0
    downs = ups.copy()
    ups[delta > 0] = delta[delta > 0]
    downs[delta < 0] = -delta[delta < 0]
    ups[ups.index[period-1]] = np.mean( ups[:period] ) #first value is sum of avg gains
    ups = ups.drop(ups.index[:(period-1)])
    downs[downs.index[period-1]] = np.mean( downs[:period] ) #first value is sum of avg losses
    downs = downs.drop(downs.index[:(period-1)])
    rs = ups.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean() / \
         downs.ewm(com=period-1,min_periods=0,adjust=False,ignore_na=False).mean()
    rsi = 100 - 100 / (1 + rs)

    # Calculate StochRSI
    stochrsi  = 100*(rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())

    return stochrsi


def get_stoch(df, period):
    df['L14'] = df['l'].rolling(window=period).min()
    # Create the "H14" column in the DataFrame
    df['H14'] = df['h'].rolling(window=period).max()
    # Create the "%K" column in the DataFrame
    df["stoch"] = 100 * ((df['c'] - df['L14']) / (df['H14'] - df['L14']))

    return df["stoch"]
