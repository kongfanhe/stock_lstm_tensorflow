# https://q.stock.sohu.com/hisHq?code=cn_601766,cn_000002&start=20000101&end=20200601

import pandas as pd
import requests
import time
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import shutil
import re


def _init_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def request_data(code, start_date, end_date):
    return _request_data_sohu(code, start_date, end_date)


def _request_data_sohu(code, start_date, end_date):
    end = end_date.strftime("%Y%m%d")
    start = start_date.strftime("%Y%m%d")
    url_base = "https://q.stock.sohu.com/hisHq?"
    cols = ['Date', 'Open', 'Close', 'Low', 'High', 'Volume']
    cols_indices = [0, 1, 2, 5, 6, 7]
    code_url = "code=" + ("cn_" if re.match(r"\d{6}", code) else "") + code
    date_url = "start=" + start + "&end=" + end
    url = url_base + code_url + "&" + date_url
    resp = requests.request("GET", url)
    resp_json = resp.json()
    if type(resp_json) is dict:
        resp_obj = resp_json
    elif type(resp_json) is list:
        resp_obj = resp.json()[0]
    else:
        raise (Exception("unknown data type..."))
    if "hq" in resp_obj:
        hq = resp_obj["hq"]
        if len(hq) > 0:
            df = pd.DataFrame(np.asarray(hq, dtype=np.str)[:, cols_indices], columns=cols)
            df = df.sort_values(by='Date', ascending=True)
            df = df.reset_index().drop("index", axis="columns")
            return df
        else:
            return None
    else:
        return None


def get_max_repeat(data):
    data = data.astype(np.float)
    flat = data.transpose().flatten()
    diff = np.diff(flat)
    change_idx = np.where(diff != 0)[0]
    repeats = np.split(flat, change_idx + 1)
    mr = max([len(u) for u in repeats])
    return mr


def download_history(data_dir, codes, start, end, min_data, max_repeat):
    start_date = datetime.strptime(start, '%Y%m%d')
    end_date = datetime.strptime(end, '%Y%m%d')
    _init_dir(data_dir)
    for c in tqdm(codes):
        df = request_data(c, start_date, end_date)
        if df is not None:
            if len(df) > min_data:
                if get_max_repeat(df.drop(columns=["Date"]).to_numpy()) < max_repeat:
                    csv_path = os.path.join(data_dir, str(c) + ".csv")
                    df.to_csv(csv_path, index=False)
                    time.sleep(0.1)


def main():
    download_history("data", ["201000", "204001"], "20100101", "20200601", 2000, 30)


if __name__ == "__main__":
    main()
