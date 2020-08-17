
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from shutil import copyfile
import shutil


def _init_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def main():
    raw_dir = os.path.join(Path().resolve().parent.parent, "data", "raw_history")

    _init_dir("clean")
    _init_dir("repeat_data")
    _init_dir("too_few")

    for f in tqdm(os.listdir(raw_dir)):
        src = os.path.join(raw_dir, f)
        df = pd.read_csv(src)
        if len(df) > 2000:
            data = df.drop(columns=["Date"]).to_numpy()
            flat = data.transpose().flatten()
            unchanged = np.split(flat, np.where(np.diff(flat) != 0)[0]+1)
            n_unchanged = max([len(u) for u in unchanged])
            if n_unchanged < 30:
                copyfile(src, os.path.join("clean", f))
            else:
                copyfile(src, os.path.join("repeat_data", f))
                print("repeat: ", n_unchanged, f)
        else:
            copyfile(src, os.path.join("too_few", f))
            # print("too few: ", len(df))


if __name__ == "__main__":
    main()
