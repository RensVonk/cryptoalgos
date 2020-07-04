import pandas as pd
import features
import matplotlib.pyplot as plt
import os
import numpy as np


def add_features(df):
    # Create features
    # df["o_scaled_50"] = df["o"].rolling(window=50, min_periods=50, axis=0).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)), raw=True)
    # df["rsi_s"] = features.get_rsi(df, period=30)
    df["rsi_f"] = features.get_rsi(df, period=14)
    df["stoch_rsi"] = features.get_stoch_rsi(df["c"], period=14)
    df["stoch"] = features.get_stoch(df, period=14)

    # We don't have to perform shifting, because we use the open price
    n_long = 2000
    df["std_short"] = df["o"].rolling(100).std()
    df["std_long"] = df["o"].rolling(n_long).std()
    mean_factor = df["std_long"].mean()/df["std_short"].mean()
    df["std_factor"] = mean_factor*df["std_short"]/df["std_long"]
    df["std_factor"] = df["std_factor"].apply(lambda x: 0.5 if x < 0.5 else x)
    df["std_factor"] = df["std_factor"].apply(lambda x: 1.5 if x > 1.5 else x)
    df["ma_fast"] = df["o"].rolling(10).mean()
    df["ma_slow"] = df["o"].rolling(40).mean()

    # min_std = df["std_factor"].min()
    # max_std = df["std_factor"].max()
    # mean_std = df["std_factor"].mean()
    #
    # # Standardize to desired interval
    # a = 0.5
    # b = 1.5
    # df["std_factor"]= df["std_factor"].apply(lambda x: (x - min_std)/(max_std - min_std))
    # df["std_factor"] = df["std_factor"].apply(lambda x: a + (b - a)*x)
    # df["std_factor"] = df["std_factor"].apply(lambda x: b if x > b else x)
    # df["std_factor"] = df["std_factor"].apply(lambda x: a if x < b else x)

    # Drop nans at beginning and end
    first_ix = df.first_valid_index()
    last_ix = df.last_valid_index()
    df = df.loc[n_long:last_ix]

    return df


def add_entry_signal(row):
    rsi_threshold = 40
    stoch_rsi_threshold = 15
    stoch_threshold = 30

    # If oversold:
    if (row["rsi_f"] < rsi_threshold) and (row["stoch_rsi"] < stoch_rsi_threshold) and (row["stoch"] < stoch_threshold) and (row["ma_fast"] > row["ma_slow"]):
        return 1
    # Else
    elif ("BTC" in row["market"]) and (row["rsi_f"] > 100 - rsi_threshold) and (row["stoch_rsi"] > 100 -  stoch_rsi_threshold) and (row["stoch"] > 100 - stoch_threshold) and (row["ma_fast"] < row["ma_slow"]):
        return  -1
    else:
        return 0


def check_exit_long(row, open_price, take_profit_pct, stop_loss_pct, spread_pct):
    return_high = (row["h"] / open_price) - 1
    return_low = (row["l"] / open_price) - 1
    # print(row, return_low, return_high)

    temp_profit = 0
    close_price = None
    if return_high >= take_profit_pct:
        temp_profit = take_profit_pct - 2*spread_pct
        close_price = round(open_price*(1+take_profit_pct), 4)

    elif return_low <= -stop_loss_pct:
        temp_profit = -(stop_loss_pct + 2*spread_pct)
        close_price = round(open_price*(1-stop_loss_pct), 4)
    return temp_profit, close_price


def check_exit_short(row, open_price, take_profit_pct, stop_loss_pct, spread_pct):
    return_high = (open_price/row["h"]) - 1 # stop loss
    return_low = (open_price/row["l"]) - 1 # Tp
    # print(row, return_low, return_high)

    temp_profit = 0
    close_price = None
    if return_high <= -stop_loss_pct:
        temp_profit = -stop_loss_pct - 2*spread_pct
        close_price = round(open_price*(1 + stop_loss_pct), 4)
    elif return_low >= take_profit_pct:
        temp_profit = take_profit_pct - 2*spread_pct
        close_price = round(open_price*(1-take_profit_pct), 4)
    return temp_profit, close_price


def perform_backtest(df, market):
    """
    Note; Inaccuracy due to
    :param df:
    :return:
    """
    take_profit_pct = 3/100
    stop_loss_pct = 4/100
    position = 0
    profit = 0
    spread_pct = 0.075/100
    profit_list = [0]
    profit_update_list = [0]
    trade_dates = [df.loc[0]["d"]]
    for ix in df.index:
        if position == 0:
            if df.loc[ix]["entry"] == 1:
                std = df.loc[ix]["std_factor"]
                position = 1
                open_price = df.loc[ix]["o"]
                print(f"Open long: {df.loc[ix]['d']} at {open_price}")

                # Check if we can close already (we open at opening bar,so we have a h/l to process)
                profit_update, close_price = check_exit_long(df.loc[ix], open_price, take_profit_pct*std, stop_loss_pct*std, spread_pct)
                if profit_update != 0:
                    profit += profit_update
                    trade_dates += [df.loc[ix]["d"]]
                    profit_list += [profit]
                    profit_update_list += [profit_update]
                    position = 0
                    print(f"Close long: {df.loc[ix]['d']} {close_price}. Trade profit: {profit_update}. Total profit: {profit}\n")
            elif df.loc[ix]["entry"] == -1:
                std = df.loc[ix]["std_factor"]
                position = -1
                open_price = df.loc[ix]["o"]
                print(f"Open short: {df.loc[ix]['d']} at {open_price}")

                # Check if we can close already (we open at opening bar,so we have a h/l to process)
                profit_update, close_price = check_exit_short(df.loc[ix], open_price, take_profit_pct*std, stop_loss_pct*std, spread_pct)
                if profit_update != 0:
                    profit += profit_update
                    trade_dates += [df.loc[ix]["d"]]
                    profit_list += [profit]
                    profit_update_list += [profit_update]
                    position = 0
                    print(f"Close short: {df.loc[ix]['d']} {close_price}. Trade profit: {profit_update}. Total profit: {profit}\n")

        # If we are long
        elif position > 0:
            # Check if we can close already (we open at opening bar,so we have a h/l to process)
            profit_update, close_price = check_exit_long(df.loc[ix], open_price, take_profit_pct*std, stop_loss_pct*std, spread_pct)
            if profit_update != 0:
                profit += profit_update
                position = 0
                trade_dates += [df.loc[ix]["d"]]
                profit_list += [profit]
                profit_update_list += [profit_update]
                print(f"Close long: {df.loc[ix]['d']} {close_price}. Trade profit: {profit_update}. Total profit: {profit}\n")
        # Short not implemented
        elif position < 0:
            # Check if we can close already (we open at opening bar,so we have a h/l to process)
            profit_update, close_price = check_exit_short(df.loc[ix], open_price, take_profit_pct * std, stop_loss_pct * std, spread_pct)
            if profit_update != 0:
                profit += profit_update
                trade_dates += [df.loc[ix]["d"]]
                profit_list += [profit]
                profit_update_list += [profit_update]
                position = 0
                print(f"Close short: {df.loc[ix]['d']} {close_price}. Trade profit: {profit_update}. Total profit: {profit}\n")
    plt.figure()
    plt.plot(trade_dates, profit_list)
    plt.grid()
    plt.xticks(rotation=90)
    plt.title(market)
    plt.tight_layout()
    plt.savefig(f"output/{market}.png")

    # return data to later aggregate
    return profit_update_list, trade_dates

# Create output dir
if not os.path.exists("output"):
    os.mkdir("output")

# Define pairs to backtest
markets = ['BTCUSDT',
           "ETHUSDT",
           "NEOUSDT",
           "XRPUSDT",
           "XMRUSDT",
           "DASHUSDT",
           "NANOUSDT",
           "BCHUSDT",
           ]

all_trade_dates = []
all_profits = []

# Iterate over all markets
for market in markets:
    # Read data and add features
    df = pd.read_csv(f"{market}.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values("d").reset_index(drop=True)
    df = add_features(df)
    df["d"] = pd.to_datetime(df["d"])
    df["market"] = market
    df["entry"] = df.apply(add_entry_signal, axis=1).shift(1)

    # Create train test_split
    cutoff_date = pd.to_datetime("2010-01-01 00:00:00")
    max_date = pd.to_datetime("2021-01-01 00:00:00")
    # df_train = df[df["d"] <= cutoff_date].reset_index(drop=True)
    df_val = df[(df["d"] > cutoff_date) & (df["d"] < max_date)].reset_index(drop=True)

    # Perform backtest
    profits, dates = perform_backtest(df_val, market)
    all_trade_dates += dates
    all_profits += profits

df = pd.DataFrame(data=np.array([all_trade_dates, all_profits]).T, columns=["d", "profit"])
df = df.sort_values("d")
df["profit_cum"] = df["profit"].cumsum()
df.plot("d", "profit_cum")
plt.grid()
plt.savefig("output/all.png")

