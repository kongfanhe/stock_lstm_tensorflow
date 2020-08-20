from downloader import request_data
import os
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import shutil
from models import LstmPredictor
from mongo_utils import update, get_target_date


def _init_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def download(data_dir, codes_raw, start_date, end_date, features):
    _init_dir(data_dir)
    codes = []
    for code in tqdm(codes_raw):
        df = request_data(code, start_date, end_date)
        if df is not None:
            df = df.filter(features)
            path = os.path.join(data_dir, code + ".csv")
            codes.append(code)
            df.to_csv(path, header=False, index=False)
    return codes


def load_to_batch(data_dir, codes, n_input):
    batch_x = []
    for code in tqdm(codes):
        p_source = os.path.join(data_dir, code + ".csv")
        df = pd.read_csv(p_source, header=None, index_col=False)
        if len(df) >= n_input:
            df = df.iloc[-n_input:, :]
        else:
            top_row = df.iloc[0, :].to_numpy()[np.newaxis, :]
            df_top = pd.DataFrame(np.repeat(top_row, n_input - len(df), axis=0))
            df = pd.concat((df_top, df))
        batch_x.append(df.to_numpy())
    batch_x = np.array(batch_x)
    return batch_x


def main():
    target_date = get_target_date()
    if target_date is not None:
        cwd = os.path.dirname(os.path.realpath(__file__))
        raw_dir = os.path.join(cwd, "_temp")
        n_inputs = [50, 100, 300]
        outputs = [[1], [2, 3, 4, 5], [30, 60, 90]]
        features = ["Close", "Volume"]
        w_files = ["weights.next.hdf5", "weights.short.hdf5", "weights.long.hdf5"]
        stocks_file = "stocks.csv"
        # stocks_file = "stocks_limited.csv"
        dates = ["+ " + str(_x) + " 天" for _output in outputs for _x in _output]
        stocks = pd.read_csv(os.path.join(cwd, stocks_file), dtype=str)
        codes_raw = stocks["code"]
        start_date = target_date - relativedelta(days=600)
        codes = download(raw_dir, codes_raw, start_date, target_date, features)
        names = stocks[stocks['code'].isin(codes)]["company_name"].tolist()
        close = np.zeros((len(codes), 0))
        volume = np.zeros((len(codes), 0))
        p_close = np.zeros((len(codes), 0))
        for n_input, output, w_file in zip(n_inputs, outputs, w_files):
            batch_x = load_to_batch(raw_dir, codes, n_input)
            predictor = LstmPredictor(n_input, len(output), len(features), os.path.join(cwd, w_file))
            pred = predictor.predict(batch_x, chunk_size=30)
            close = np.concatenate((close, pred[:, :, features.index("Close")]), axis=1)
            volume = np.concatenate((volume, pred[:, :, features.index("Volume")]), axis=1)
            p_close = batch_x[:, -1, features.index("Close")]
        sec_names = ["上证指数", "深证成指", "创业板指"]
        sec_codes = ["zs_000001", "zs_399001", "zs_399006"]
        download(raw_dir, sec_codes, start_date, target_date, features)
        batch_x = load_to_batch(raw_dir, sec_codes, 50)
        predictor = LstmPredictor(50, 1, len(features), file=os.path.join(cwd, "weights.sector.hdf5"))
        pred = predictor.predict(batch_x, chunk_size=30)
        sec_close = pred[:, :, features.index("Close")]
        sec_p_close = batch_x[:, -1, features.index("Close")]
        update(codes, names, dates, close, volume, p_close, sec_names, sec_close, sec_p_close, target_date)
    else:
        print("no need to update")


if __name__ == "__main__":
    main()
