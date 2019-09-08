import logging
import os
import numpy as np
from keras.callbacks import LambdaCallback, EarlyStopping

from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from src.datapreprocessing.SequencesGenerator import DataGenerator
from src.datapreprocessing.dataset_preprocessing import generate_tokens, generate_sequences
from src.model.model import get_model, sample
from src.model.plotting import plot_train_validation_loss

seq_len = 20
start_token = ('| ' * seq_len)
input_file = "data/poems.csv"
embedding_size = 100
units = 256
epochs = 100
n_sample = None


def generate_poems(seed_text, next_words, model, max_sequence_len, temp):
    output = seed_text
    seed_text = start_token + seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample(probs, temperature=temp)
        output_word = reverse_word_map[y_class] if y_class > 0 else ''
        if output_word == "|":
            break
        seed_text += output_word + ' '
        output += output_word + ' '
    return output


def on_epoch_end(epoch, _):
    examples_file.write('\n----- Generating text after Epoch: {}\n'.format(epoch))
    for t in [0.2, 0.4, 0.6, 0.8]:
        examples_file.write('\n----- Diversity:' + str(t) + '\n')
        examples_file.write(generate_poems('', 400, model, 15, t))


if __name__ == "__main__":

    if not os.path.isdir('./examples/'):
        os.makedirs('./examples/')

    if not os.path.isdir('./logs/'):
        os.makedirs('./logs/')


    tokens, index, count, tokenizer = generate_tokens(input_file, n_sample=n_sample, seq_len=seq_len)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    print(tokens)

    X, y, n_seq = generate_sequences(tokens, seq_len)
    print(X[1], y[1])

    generator_params = {
            'batch_size': 64,
            'seq_len': seq_len,
            'n_classes': count,
            'shuffle': True}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    training_generator = DataGenerator(X_train, y_train, **generator_params)
    validation_generator = DataGenerator(X_test, y_test, **generator_params)

    model = get_model(count, embedding_size=embedding_size, lstm_units=units)

    opti = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opti)


    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    print("Generating model...")
    examples_file = open("./examples/example.txt", "w")
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  use_multiprocessing=True,
                                  callbacks=[print_callback, early_stopping],
                                  workers=6)

    plot_train_validation_loss(history.history['loss'], history.history['val_loss'])

