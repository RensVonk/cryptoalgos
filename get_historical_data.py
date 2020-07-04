import pandas as pd
import requests
import time
from datetime import datetime
pd.set_option("display.max_columns", 99)

markets = ['BTC',
           "ETH",
           "LINK",
           "XMR",
           "DASH",
           "NANO"
           ]
markets = [m + "USDT" for m in markets]
tick_interval = '30m'
market_to_df = {}

for market in markets:
    print(f"Getting data for {market}\n")
    end_time = int(time.time() * 1000)
    df_list = []
    for i in range(50):
        # Update dates
        start_time = int((end_time / 1000 - 30 * 500 * 60) * 1000)
        print(datetime.utcfromtimestamp(start_time / 1000), datetime.utcfromtimestamp(end_time / 1000))

        url = f"https://api.binance.com/api/v1/klines?symbol={market}&interval={tick_interval}&startTime={start_time}&endTime={end_time}"
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        df = df.rename(columns={0:"d",
                                1:"o",
                                2:"h",
                                3:"l",
                                4:"c",
                                5:"v"})

        try:
            df = df[["d","o","h","l","c","v"]]
            df["d"] = df["d"].apply(lambda x: datetime.utcfromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
            df_list += [df]

            end_time = start_time
        except:
            pass

    # Process
    df = pd.concat(df_list, axis=0).reset_index(drop=True)
    df.sort_values("d")
    df.to_csv(f"{market}.csv", index=False)
