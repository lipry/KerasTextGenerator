from keras import Model
import matplotlib.pyplot as plt
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from src.datapreprocessing.dataset_preprocessing import generate_tokens, generate_sequences

# parameters
imput_file = "data/poems.csv"
seq_len = 20
n_sample = 50
embedding_size = 100
units = 256
epochs = 10
batch_size = 32


def get_model(input_dim, embedding_size=100, lstm_units=256):
    model_in = Input(shape=(None, ))
    x = Embedding(input_dim, embedding_size)(model_in)
    x = LSTM(lstm_units)(x)
    x = Dropout(0.2)(x)
    model_out = Dense(input_dim, activation='softmax')(x)

    model = Model(model_in, model_out)
    return model


tokens, index, count = generate_tokens(imput_file, sample=n_sample, seq_len=seq_len)
X, y, n_seq = generate_sequences(tokens, count, seq_len)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = get_model(count, embedding_size=embedding_size, lstm_units=units)

opti = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opti)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

print('\nhistory dict:', history.history)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('data/graph/books_read.png')

print('\n# Evaluate on test data')
results = model.evaluate(X_test, y_test, batch_size=64)
print('test loss, test acc:', results)

