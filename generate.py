import multiprocessing as mpc
import os
import pandas as pd
import numpy as np
import random
import shutil
from tqdm import tqdm
from dateutil.relativedelta import relativedelta


random.seed(0)


def create(batch_dir, raw_dir, batch_size, features, input_n, output_days, w_day, rand):
    paths = []
    counts = []
    for f in tqdm(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, f)
        paths.append(p)
        dates = pd.Series(pd.to_datetime(_read_raw_csv(p)['Date'], format='%Y-%m-%d'))
        max_n = _max_index_date(dates, output_days, w_day)
        c = max_n - input_n
        counts.append(c)
    _idx = list(range(np.sum(counts) // batch_size)) * batch_size
    if rand:
        random.shuffle(_idx)
    indices = np.array_split(_idx, np.cumsum(counts))[:-1]
    _init_dir(batch_dir)
    q = mpc.Manager().Queue()
    pool = mpc.Pool()
    pool.apply_async(_listener, (q,))
    jobs = []
    for p, idx in zip(paths, indices):
        jobs.append(pool.apply_async(_save_one, (p, idx, features, input_n, output_days, batch_dir, w_day, q)))
        # _save_one(p, idx, features, input_n, output_days, batch_dir, w_day)
    [job.get() for job in jobs]
    q.put("kill")
    pool.close()
    pool.join()


def split_train_test(src_dir, train_dir, test_dir, train_size):
    train_num = round(train_size * len(os.listdir(src_dir)))
    m = 0
    _init_dir(train_dir)
    _init_dir(test_dir)
    for f in os.listdir(src_dir):
        if m < train_num:
            shutil.move(os.path.join(src_dir, f), os.path.join(train_dir, f))
        else:
            shutil.move(os.path.join(src_dir, f), os.path.join(test_dir, f))
        m += 1
    shutil.rmtree(src_dir)


def _save_one(stock_path, batch_indices, features, input_n, output_days, batch_dir, w_day, queue=None):
    """
    output_day = 0, i.e. the first predicted day
    """
    df = _read_raw_csv(stock_path)
    dates = pd.Series(pd.to_datetime(df['Date'], format='%Y-%m-%d'))
    arr = df.filter(features).to_numpy()
    max_n = _max_index_date(dates, output_days, w_day)
    for bi, i in zip(batch_indices, range(input_n, max_n)):
        if w_day:
            ny = [i + d for d in output_days]
        else:
            ny = [np.sum(dates < (dates.iloc[i] + relativedelta(days=d))) for d in output_days]
        _x = np.reshape(arr[i - input_n:i, :], -1)
        _y = np.reshape(arr[ny, :], -1)
        b_file = os.path.join(batch_dir, "batch_" + str(bi) + ".csv")
        line = ",".join(np.concatenate((_x, _y)).astype(str)) + "\n"
        if queue is None:
            open(b_file, "a").write(line)
        else:
            queue.put(b_file + "\t" + line)


def _max_index_date(dates, output_days, w_day):
    if w_day:
        max_n = len(dates) - max(output_days)
    else:
        latest_possible = dates.iloc[-1] - relativedelta(days=max(output_days))
        max_n = np.sum(dates <= latest_possible)
    return max_n


def _listener(queue):
    print("listener started...")
    m = 0
    while True:
        message = queue.get()
        if message == 'kill':
            print("listener killed...")
            break
        else:
            m = m + 1
            if m % 10000 == 0:
                print("saving data : ", m)
            file_path, line = message.split("\t")
            open(file_path, "a").write(line)


def _init_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def _read_raw_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values(by='Date', ascending=True)
    df = df.reset_index().drop("index", axis="columns")
    return df
