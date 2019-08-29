import logging
from keras import Model
import numpy as np
from keras.callbacks import LambdaCallback
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from src.datapreprocessing.dataset_preprocessing import generate_tokens, generate_sequences
from src.model_validation.plotting import plot_train_validation_loss


if __name__ == "__main__":
    logger = logging.getLogger('TextGenerator')
    logger.setLevel(logging.DEBUG)

    # logs file handler
    fh = logging.FileHandler('text_gen.log')
    fh.setLevel(logging.DEBUG)
    # logs console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(fh)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    seq_len = 20
    start_token = ('| ' * seq_len)
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


    tokens, index, count, tokenizer = generate_tokens(imput_file, sample=n_sample, seq_len=seq_len)
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

    X, y, n_seq = generate_sequences(tokens, count, seq_len)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = get_model(count, embedding_size=embedding_size, lstm_units=units)

    opti = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opti)


    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


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
        logger.debug("------  text after epoch {}  ------".format(epoch))
        for t in [0.2, 0.5, 1.0, 1.2]:
            logger.debug('----- temperature: {}'.format(t))
            logger.debug(generate_poems('', 400, model, 15, t))


    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[print_callback])

    plot_train_validation_loss(history.history['loss'], history.history['val_loss'])

    print('\n# Evaluate on test data')
    results = model.evaluate(X_test, y_test, batch_size=64)
    print('test loss, test acc:', results)

