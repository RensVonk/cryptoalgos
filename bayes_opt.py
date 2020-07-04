import pandas as pd
import features
import os
import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe

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
           #"ETHUSDT",
           #"LINKUSDT",
           #"XMRUSDT",
           #"DASHUSDT",
           #"NANOUSDT"
           ]

# Set up space dictionary with specified hyperparameters
space = {'RSI_list': hp.quniform('RSI_list', 25, 45, 5), 'StochRSI_list': hp.quniform('StochRSI_list', 5, 35, 5),
         'Stoch_list': hp.quniform('Stoch_list', 5, 35, 5), 'TP_list': hp.quniform('TP_list', 1, 10, 1),
         'SL_list': hp.quniform('SL_list', 1, 10, 1)}

for market in markets:
    def objective(space):
        df = pd.read_csv(f"{market}.csv")
        df = df.drop_duplicates().reset_index(drop=True)
        df = df.sort_values("d").reset_index(drop=True)
        df = add_features(df)
        df["d"] = pd.to_datetime(df["d"])
        df["market"] = market
        #introduce hyperparameters
        df["entry"] = df.apply(add_entry_signal, args=(space['RSI_list'], space['StochRSI_list'], space['Stoch_list']), axis=1).shift(1)
        # Create train test_split
        cutoff_date = pd.to_datetime("2010-01-01 00:00:00")
        max_date = pd.to_datetime("2021-01-01 00:00:00")
        # df_train = df[df["d"] <= cutoff_date].reset_index(drop=True)
        df_val = df[(df["d"] > cutoff_date) & (df["d"] < max_date)].reset_index(drop=True)
        # Perform backtest
        success_percentage, profit, sharpe = perform_backtest(df_val, market, space['SL_list'], space['TP_list'])
        loss = sharpe * -1
        return loss

    best = fmin(fn=objective,space=space, max_evals=100, rstate=np.random.RandomState(42), algo=tpe.suggest)
    print(best)
