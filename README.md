# Tensorflow LSTM stock predictor

## Introduction

A LSTM model that predict future **N day price and volume** based on the past **M day price and volume**.

The LSTM model is specialized into four variant models:
* next-day composite index predictor
* next-day stock predictor
* future-5-days stock predictor
* future-30-days stock predictor

These four models share the same model definition, but each has a unique network architecture and weights, should be trained seperately.

## How to use
1. Install Python >= 3.6
2. (Optionally) Install MongoDB to persist everyday stock prediction.
3. Install necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    **note**: if you do not need MongoDB for data persistence, remove this line 
    ```
    pymongo==3.10.1
    ```
    from *requirements.txt*
4. Download stock history data and train the four LSTM models.
    ```bash
    python train.py
    ```
    The training process will download Shanghai and Shenzhen stock market history from [Sohu Finance Api](https://q.stock.sohu.com/), and save the data to *.csv* format.

    After downloading it will train four models sequentially, generating four weight files:
    * weights.next.hdf5
    * weights.short.hdf5
    * weights.long.hdf5
    * weights.sector.hdf5
5. (Optinally) Update the stock prediction based on the newest data available (till the most recent trading day), and persist the prediction to MongoDB.
    ```bash
    python update.py
    ```


