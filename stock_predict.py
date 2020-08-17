from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from tqdm import tqdm
import os
from tensorflow.keras.callbacks import ModelCheckpoint


class LstmPredictor:

    def __init__(self, n_input, n_output, n_feature, file=None):
        model = Sequential()
        model.add(LSTM(n_input, return_sequences=True))
        model.add(LSTM(n_input * n_feature, return_sequences=False))
        model.add(Dense(n_input * n_feature * 10))
        model.add(Dense(n_input * n_feature * 5))
        model.add(Dense(n_output * n_feature))
        model.add(tf.keras.layers.Reshape((n_output, n_feature)))
        self.model: Sequential = model
        if file is not None:
            self.model: Sequential = keras.models.load_model(file)
        self.n_output = n_output
        self.n_feature = n_feature

    def train(self, train_seq, val_data, epochs, model_path):
        ck_pt = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(x=train_seq, validation_data=val_data, epochs=epochs, callbacks=[ck_pt], verbose=1)

    def predict(self, x, chunk_size=None):
        """
        x.shape : [batch_size, time_steps, features]
        """
        scale_max = np.max(x, axis=1)[:, np.newaxis, :]
        scale_min = np.min(x, axis=1)[:, np.newaxis, :]
        scale_range = np.maximum(scale_max - scale_min, np.finfo(float).eps)
        x_scaled = (x - scale_min) / scale_range
        total_size = len(x)
        if chunk_size is None:
            chunk_size = total_size
        indices = list(np.arange(0, total_size, chunk_size))
        if total_size != indices[-1]:
            indices = indices + [total_size]
        y_scaled = np.zeros((0, self.n_output, self.n_feature))
        print("Predicting per chunk:")
        for i in tqdm(range(len(indices) - 1)):
            n1, n2 = indices[i], indices[i+1]
            _y = self.model.predict(x_scaled[n1:n2, :, :])
            y_scaled = np.concatenate((y_scaled, _y), axis=0)
        y = y_scaled * scale_range + scale_min
        return y


class DataSequence(keras.utils.Sequence):

    def __init__(self, batch_dir, batch_size, n_input, n_output, n_feature):
        self.length = len(os.listdir(batch_dir))
        self.paths = []
        for f in os.listdir(batch_dir):
            self.paths.append(os.path.join(batch_dir, f))
        self.n_feature = n_feature
        self.n_input = n_input
        self.n_output = n_output
        self.batch_size = batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, batch_index):
        df = pd.read_csv(self.paths[batch_index], header=None)
        arr = df.to_numpy()
        nfe = self.n_feature
        nou = self.n_output
        nin = self.n_input
        batch_x = np.reshape(arr[:, :nin * nfe], (self.batch_size, nin, nfe))
        batch_y = np.reshape(arr[:, nin * nfe:], (self.batch_size, nou, nfe))
        scale_max = np.max(batch_x, axis=1)[:, np.newaxis, :]
        scale_min = np.min(batch_x, axis=1)[:, np.newaxis, :]
        batch_x = (batch_x - scale_min) / (scale_max - scale_min)
        batch_y = (batch_y - scale_min) / (scale_max - scale_min)
        return batch_x, batch_y

    def convert_to_data(self):
        print("Sequence: converting to data")
        x, y = [], []
        for i in tqdm(range(self.__len__())):
            batch_x, batch_y = self.__getitem__(i)
            x.extend(batch_x)
            y.extend(batch_y)
        x, y = np.asarray(x), np.asarray(y)
        return x, y
