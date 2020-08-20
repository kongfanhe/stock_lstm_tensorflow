from models import DataSequence, LstmPredictor
from downloader import download_history
import generate as gen
import pandas as pd


def train_model(net_type, data_dir, batch_size, epochs):
    features = ["Close", "Volume"]
    model_path = "weights." + str(net_type) + ".hdf5"
    if net_type == "next":
        input_n, output_days, w_day, rand = 50, [0], True, True
    elif net_type == "short":
        input_n, output_days, w_day, rand = 100, [1, 2, 3, 4], True, True
    elif net_type == "long":
        input_n, output_days, w_day, rand = 300, [30, 60, 90], False, True
    else:
        input_n, output_days, w_day, rand = 50, [0], True, True
    gen.create("_temp", data_dir, batch_size, features, input_n, output_days, w_day, rand)
    gen.split_train_test("_temp", "batch_train", "batch_val", train_size=0.9)
    train_seq = DataSequence("batch_train", batch_size, input_n, len(output_days), len(features))
    val_seq = DataSequence("batch_val", batch_size, input_n, len(output_days), len(features))
    val_x, val_y = val_seq.convert_to_data()
    predictor = LstmPredictor(input_n, len(output_days), len(features))
    predictor.train(train_seq, (val_x, val_y), epochs, model_path)
    print(net_type, " all done")


def download_data(sector_dir, stock_dir, sector_codes, stock_codes, start, end, min_data, max_repeat):
    download_history(sector_dir, sector_codes, start, end, min_data, max_repeat)
    download_history(stock_dir, stock_codes, start, end, min_data, max_repeat)


def main():
    # batch_size, epochs, start, end, min_data = 2, 1, "20100101", "20120101", 300
    # stock_file, sector_codes = "stocks_limited.csv", ["zs_000001"]

    batch_size, epochs, start, end, min_data = 200, 5, "20100101", "20200601", 2000
    stock_file, sector_codes = "stocks.csv", ["zs_000001", "zs_399001", "zs_399006"]

    stock_dir, sector_dir = "data_stock", "data_sector"
    stock_codes = pd.read_csv(stock_file, dtype=str)["code"].values
    download_data(sector_dir, stock_dir, sector_codes, stock_codes, start, end, min_data, 30)
    train_model("sector", sector_dir, batch_size, epochs)
    train_model("long", stock_dir, batch_size, epochs)
    train_model("short", stock_dir, batch_size, epochs)
    train_model("next", stock_dir, batch_size, epochs)


if __name__ == "__main__":
    main()
