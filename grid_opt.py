import pandas as pd
import features
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
    mean_factor = df["std_long"].mean() / df["std_short"].mean()
    df["std_factor"] = mean_factor * df["std_short"] / df["std_long"]
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


def add_entry_signal(row, RSI, StochRSI, Stoch):
    rsi_threshold = RSI
    stoch_rsi_threshold = StochRSI
    stoch_threshold = Stoch
    # If oversold:
    if (row["rsi_f"] < rsi_threshold) and (row["stoch_rsi"] < stoch_rsi_threshold) and (
            row["stoch"] < stoch_threshold):
        return 1
    else:
        return 0


def check_exit_long(row, open_price, take_profit_pct, stop_loss_pct, spread_pct):
    return_high = (row["h"] / open_price) - 1
    return_low = (row["l"] / open_price) - 1
    # print(row, return_low, return_high)

    temp_profit = 0
    close_price = None
    result_right = 0
    result_wrong = 0
    if return_high >= take_profit_pct:
        temp_profit = take_profit_pct - 2 * spread_pct
        close_price = round(open_price * (1 + take_profit_pct), 4)
        result_right = 1
    elif return_low <= -stop_loss_pct:
        temp_profit = -(stop_loss_pct + 2 * spread_pct)
        close_price = round(open_price * (1 - stop_loss_pct), 4)
        result_wrong = 1
    return temp_profit, close_price, result_right, result_wrong


def perform_backtest(df, market, tp, sl):
    """
    Note; Inaccuracy due to
    :param df:
    :return:
    """
    take_profit_pct = tp / 100
    stop_loss_pct = sl / 100
    position = 0
    right = 0
    wrong = 0
    profit_list = [0]
    spread_pct = 0.075 / 100
    profit = 0
    for ix in df.index:
        if position == 0:
            if df.loc[ix]["entry"] == 1:
                position = 1
                open_price = df.loc[ix]["o"]
                # Check if we can close already (we open at opening bar,so we have a h/l to process)
                profit_update, close_price, result_right, result_wrong = check_exit_long(df.loc[ix], open_price, take_profit_pct,
                                                             stop_loss_pct, spread_pct)
                if profit_update != 0:
                    position = 0
                    profit_list += [profit_update]
                    profit += profit_update
                    right += result_right
                    wrong += result_wrong
        # If we are long
        elif position > 0:
            # Check if we can close already (we open at opening bar,so we have a h/l to process)
            profit_update, close_price, result_right, result_wrong = check_exit_long(df.loc[ix], open_price, take_profit_pct,
                                                         stop_loss_pct, spread_pct)
            if profit_update != 0:
                position = 0
                profit_list += [profit_update]
                profit += profit_update
                right += result_right
                wrong += result_wrong
    success_percentage = right / (right + wrong)
    sharpe = np.mean(profit_list) / np.std(profit_list)

    return success_percentage, profit, sharpe


# Create output dir
if not os.path.exists("output"):
    os.mkdir("output")

# Define pairs to backtest
markets = ["BTCUSDT",
           "ETHUSDT",
           "LINKUSDT",
           "XMRUSDT",
           "DASHUSDT",
           "NANOUSDT"
           ]

RSI_list = range(25, 45, 5)
StochRSI_list = range(5, 35, 5)
Stoch_list = range(5, 35, 5)
TP_list = range(2, 8, 1)
SL_list = range(2, 8, 1)

# Iterate over all markets
for market in markets:
    # Read data and add features
    success_high = 0
    sharpe_high = 0
    profit_high = 0
    RSI = 0
    StochRSI = 0
    Stoch = 0
    TP = 0
    SL = 0
    df = pd.read_csv(f"{market}.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.sort_values("d").reset_index(drop=True)
    df = add_features(df)
    df["d"] = pd.to_datetime(df["d"])
    df["market"] = market
    for a in RSI_list:
        for b in StochRSI_list:
            for c in Stoch_list:
                for d in TP_list:
                    for e in SL_list:
                        df["entry"] = df.apply(add_entry_signal, args=(a, b, c), axis=1).shift(1)
                        # Create train test_split
                        cutoff_date = pd.to_datetime("2010-01-01 00:00:00")
                        max_date = pd.to_datetime("2021-01-01 00:00:00")
                        # df_train = df[df["d"] <= cutoff_date].reset_index(drop=True)
                        df_val = df[(df["d"] > cutoff_date) & (df["d"] < max_date)].reset_index(drop=True)

                        # Perform backtest
                        success_percentage, profit, sharpe = perform_backtest(df_val, market, d, e)
                        print(f"Check for {market}: RSI = {a}, StochRSI = {b}, Stochastic = {c}, take profit {d} and stop loss {e}")
                        print(f"With Sharpe ratio = {round(sharpe, 4)} and profit = {round(profit, 4)}")
                        if sharpe > sharpe_high:
                            profit_high = profit
                            sharpe_high = sharpe
                            success_high = success_percentage
                            RSI = a
                            StochRSI = b
                            Stoch = c
                            TP = d
                            SL = e
                            print(f"UPDATE SHARPE HIGH")
    print()
    print(f"FINAL METRICS")
    print(f"Optimal oversold metrics of {market} for highest Sharpe ratio: RSI = {RSI}, StochRSI = {StochRSI} and Stoch = {Stoch}")
    print(f"Optimal take profit = {TP} and stop loss = {SL}")
    print(f"with Sharpe = {round(sharpe_high,4)} and profit = {round(profit_high,4)}")
    print()
    #Benchmark for profit and sharpe
    normal_return = df["o"].pct_change()
    normal_return = normal_return.drop(normal_return.index[0])
    market_sharpe = np.mean(normal_return) / np.std(normal_return)
    bench_return = (df.iloc[-1,1]-df.iloc[0,1])/df.iloc[0,1]
    print(f"Benchmark Sharpe ratio of {market} = {market_sharpe} and return = {bench_return}")
    print()
