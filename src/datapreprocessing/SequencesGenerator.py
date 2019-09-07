import numpy as np
from keras.utils import Sequence, to_categorical


class DataGenerator(Sequence):
    def __init__(self, sequences, y, batch_size=32, seq_len=20, n_classes=10, shuffle=True):
        self.batch_size = batch_size
        self.labels = y
        self.sequences = sequences
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idx_sequence_temp):
        X = np.empty((self.batch_size, self.seq_len))
        y = np.empty(self.batch_size, dtype=int)
        for i, seq_idx in enumerate(idx_sequence_temp):
            X[i, ] = self.sequences[seq_idx]
            y[i] = self.labels[seq_idx]
        return X, to_categorical(y, num_classes=self.n_classes)