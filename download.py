# https://q.stock.sohu.com/hisHq?code=cn_601766,cn_000002&start=20000101&end=20200601

import pandas as pd
import requests
import time
import numpy as np
import os
from tqdm import tqdm


def main():
    stocks = pd.read_csv("stocks.csv", dtype=str)
    codes = stocks["code"].values
    columns = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']
    columns_index = [0, 1, 2, 5, 6, 7]
    start = "20100101"
    end = "20200620"
    for c in tqdm(codes):
        url = "https://q.stock.sohu.com/hisHq?"
        url += "code=cn_" + str(c)
        url += "&start=" + start + "&end=" + end
        resp = requests.request("GET", url)
        resp_json = resp.json()
        if type(resp_json) is dict:
            resp_obj = resp_json
        elif type(resp_json) is list:
            resp_obj = resp.json()[0]
        else:
            print(c)
            resp_obj = None
        if "hq" in resp_obj:
            arr = np.asarray(resp_obj["hq"])
            df = pd.DataFrame(arr[:, columns_index], columns=columns)
            csv_path = os.path.join("data", "history_" + str(c) + ".csv")
            df.to_csv(csv_path, index=False)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
