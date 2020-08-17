from stock_predict import DataSequence, LstmPredictor
import generate as gen

import os
from pathlib import Path


def main(net_type):

    if net_type == "sector":
        data_dir = os.path.join(Path().resolve().parent.parent, "data", "sector_index")
    else:
        data_dir = os.path.join(Path().resolve().parent.parent, "data", "stocks_20200707")
        # data_dir = "raw_stocks_data"

    batch_size, epochs = 200, 5
    
    features = ["Close", "Volume"]
    model_path = "weights." + str(net_type) + ".best.hdf5"

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


if __name__ == "__main__":
    main("sector")
    main("long")
    main("short")
    main("next")
