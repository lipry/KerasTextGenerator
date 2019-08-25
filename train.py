from keras import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import RMSprop

from src.datapreprocessing.dataset_preprocessing import generate_tokens, generate_sequences

# parameters
seq_len = 20
n_sample = 100
embedding_size = 100
units = 256
epochs = 100
batch_size = 32


def get_model(input_dim, embedding_size=100, lstm_units=256):
    model_in = Input(shape=(None, ))
    x = Embedding(input_dim, embedding_size)(m)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.2)(x)
    model_out = Dense(input_dim, activation='softmax')(x)

    model = Model(model_in, model_out)
    return model


tokens, index, count = generate_tokens("data/poems.csv", sample=n_sample, seq_len=seq_len)
X, y, n_seq = generate_sequences(tokens, count, seq_len)

model = get_model(seq_len, embedding_size=embedding_size, lstm_units=units)

opti = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opti)
model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True)
