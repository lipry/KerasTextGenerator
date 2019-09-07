import numpy as np
from keras import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def get_model(input_dim, embedding_size=100, lstm_units=256):
    model_in = Input(shape=(None, ))
    x = Embedding(input_dim, embedding_size)(model_in)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.2)(x)
    model_out = Dense(input_dim, activation='softmax')(x)

    model = Model(model_in, model_out)
    return model