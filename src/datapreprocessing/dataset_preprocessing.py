import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils


def generate_tokens(filename, char_level=False, sample=100, seq_len=20):
    df = preprocessing(filename, n=sample)

    start_token = ('| ' * seq_len)
    texts = '\n'.join((start_token + " " + df['text']).tolist())
    texts = re.sub(r'([.,:;!?])', r' \1 ', texts)
    texts = re.sub(r'(\n)', r' \1 ', texts)
    tokenizer = Tokenizer(char_level=char_level, filters='')
    tokenizer.fit_on_texts([texts])

    tokens = tokenizer.texts_to_sequences([texts])[0]

    return tokens, tokenizer.word_index, len(tokenizer.word_index)+1


def preprocessing(filename, sep=";", n=100):
    poems = pd.read_csv(filename, sep=sep).sample(n=n)
    poems.text = poems.text.astype(str)
    poems['text'] = poems['text'].map(lambda x: x if type(x) != str else x.lower())
    return poems


def generate_sequences(tokens, token_len, seq_len):
    X = []
    y = []
    for i in range(0, len(tokens) - seq_len):
        X.append(tokens[i:i+seq_len])
        y.append(tokens[i + seq_len])

    y = np_utils.to_categorical(y, num_classes=token_len)

    return X, y, len(X)
