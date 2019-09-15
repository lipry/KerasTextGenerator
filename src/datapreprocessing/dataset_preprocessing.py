import re
import pandas as pd
from keras.preprocessing.text import Tokenizer


def preprocessing(filename, sep=";"):
    poems = pd.read_csv(filename, sep=sep)
    poems.text = poems.text.astype(str)
    poems['text'] = poems['text'].map(lambda x: x if type(x) != str else x.lower())
    return poems


def generate_tokens(filename, n_sample=None, char_level=False, seq_len=20):
    df = preprocessing(filename)
    if n_sample is not None:
        df = df.sample(n=n_sample)

    start_token = "<s>"
    texts = '\n'.join((start_token + " " + df['text']).tolist())
    texts = re.sub(r'([.,:;!?])', r' \1 ', texts)
    texts = re.sub(r'(\n)[\n]*', r' \1 ', texts)
    tokenizer = Tokenizer(char_level=char_level, filters='')
    tokenizer.fit_on_texts([texts])

    tokens = tokenizer.texts_to_sequences([texts])[0]

    return tokens, tokenizer.word_index, len(tokenizer.word_index)+1, tokenizer


def generate_sequences(tokens, seq_len):
    X = []
    y = []
    for i in range(0, len(tokens) - seq_len):
        X.append(tokens[i:i+seq_len])
        y.append(tokens[i+seq_len])
    return X, y, len(X)



