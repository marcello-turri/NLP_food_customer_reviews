import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

PATH = 'Restaurant_Reviews.tsv'
PATH_SW = 'englishST.txt'
NUM_WORDS_LENGTH = 10000

def get_num_words():
    return NUM_WORDS_LENGTH

def import_dataset(path):
    return pd.read_csv(path, delimiter='\t')

def text_label_split(df):
    return df['Review'], df['Liked']

def train_test_split(text, label, test_size=0.2):
    train_text = text[int(len(text)*test_size):]
    train_label = label[int(len(label)*test_size):]
    test_text = text[:int(len(text) * test_size)]
    test_label = label[:int(len(label) * test_size):]
    return train_text, train_label, test_text, test_label

def read_stopwords(path):
    with open(path, 'r') as f:
        content = f.read()
        stopwords = content.split('\n')
    f.close()
    return stopwords

def remove_stopwords(val):
    stopwords = read_stopwords(PATH_SW)
    words_in_sentence = val.split()
    new_sentence = []
    for word in words_in_sentence:
        if word not in stopwords:
            new_sentence.append(word)
    return ' '.join(new_sentence)

def to_lower_case(val):
    return val.lower()

def remove_punctuation(val):
    translator = val.maketrans('', '', string.punctuation)
    text_without_punctuation = val.translate(translator)
    return text_without_punctuation

def tokenize(train_text, test_text):
    tokenizer = Tokenizer(num_words=NUM_WORDS_LENGTH,
                          oov_token='<OOV>')
    tokenizer.fit_on_texts(train_text)
    train_sequences = tokenizer.texts_to_sequences(train_text)
    test_sequences = tokenizer.texts_to_sequences(test_text)
    train_sequences = pad_sequences(train_sequences, padding='post', truncating='post')
    test_sequences = pad_sequences(test_sequences, padding='post', truncating='post')
    return np.array(train_sequences), np.array(test_sequences)

def pipeline():
    df = import_dataset(PATH)
    text, label = text_label_split(df)
    text = text.apply(to_lower_case)
    text = text.apply(remove_punctuation)
    text = text.apply(remove_stopwords)
    train_text, train_label, test_text, test_label = train_test_split(text, label, test_size=0.2)
    train_sequences, test_sequences = tokenize(train_text, test_text)
    return train_sequences, train_label, test_sequences, test_label

def getting_sentences():
    df = import_dataset(PATH)
    text, label = text_label_split(df)
    text = text.apply(to_lower_case)
    text = text.apply(remove_punctuation)
    text = text.apply(remove_stopwords)
    train_text, train_label, test_text, test_label = train_test_split(text, label, test_size=0.2)
    return train_text, train_label, test_text, test_label
