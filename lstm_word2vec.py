# coding: utf-8

MODEL_NAME = "lstm_word2vec"
TRAIN_DATASETS = ["data/test_imdb.csv", "data/train_imdb.csv", "data/test_rt_en.csv", "data/train_rt_en.csv"]
TOKENIZER_NAME = "lstm_word2vec_tokenizer"

RANDOM_SEED = 42

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 70

LSTM_DIM = 128
BIDIRECTIONAL = True
EMBEDDING_DIM = 300
DROPOUT_U = 0.2
DROPOUT_W = 0.2
DROPOUT_BEFORE_LSTM = 0.2
DROPOUT_AFTER_LSTM = 0.2

MAX_EPOCHES = 2
BATCH_SIZE = 128


import pickle

import pandas as pd
import numpy as np

np.random.seed(RANDOM_SEED)

from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from gensim.models import Word2Vec



## Load data

def load_data(data_files, test_size=0.1):
    datasets = []
    for dataset in data_files:
        datasets.append(pd.read_csv(dataset, sep="|"))
    whole_data = pd.concat(datasets)
    return train_test_split(whole_data, test_size=test_size)

train_data, test_data = load_data(TRAIN_DATASETS)
print("Data loaded")

## Preprocess

negatives = {
    "didn't": "didn_`_t",
    "couldn't": "couldn_`_t",
    "can't": "can_`_t",
    "don't": "don_`_t",
    "wouldn't": "wouldn_`_t",
    "doesn't": "doesn_`_t",
    "wasn't": "wasn_`_t",
    "weren't": "weren_`_t",
    "shouldn't":"shouldn_`_t",
    "isn't": "isn_`_t",
    "aren't": "aren_`_t",
}

def preprocess(text):
    text = text.lower()
    text = text.replace('<br />', ' ')
    text = ' '.join(tweet_tokenizer.tokenize(text))
    for k, v in negatives.items():
        text = text.replace(k, v)
    return text

train_data['prep_text'] = train_data['text'].map(preprocess)
test_data['prep_text'] = test_data['text'].map(preprocess)
print("Text preprocessed")

## Tokenize

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='"#$%&()*+-/:;<=>@[\\]^{|}~\t\n,.')
tokenizer.fit_on_texts(train_data['prep_text'])
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index))
print('Found %s unique tokens.' % len(word_index))

with open(TOKENIZER_NAME,'wb') as ofile:
    pickle.dump(tokenizer, ofile)
    ofile.close()

sequences_train = tokenizer.texts_to_sequences(train_data['prep_text'])
sequences_test = tokenizer.texts_to_sequences(test_data['prep_text'])

x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
y_train = train_data['label']
y_test= test_data['label']
print("Text tokenized")

## Get word2vec embeddings

print("Get word2vec embeddings")

word2vec_google = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
word2vec_google.init_sims(replace=True)
print("Loaded")

def get_embedding(word2vec_model, word):
    try:
        return word2vec_model[word]
    except KeyError:
        return np.zeros(word2vec_model.vector_size)

    
embedding_weights_google = np.zeros((nb_words, word2vec_google.vector_size))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_weights_google[i] = get_embedding(word2vec_google, word)


print("Embeddings matrix created")


## Create model

def create_model(pretrained_embedding_weights=None, bidirectional=False):
    model = Sequential()
    if pretrained_embedding_weights is not None:
        model.add(Embedding(nb_words,
                            EMBEDDING_DIM, 
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False,
                            weights=[pretrained_embedding_weights]))
    else:
        model.add(Embedding(nb_words, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(DROPOUT_BEFORE_LSTM))
    if bidirectional:
        model.add(Bidirectional(LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W)))
    else:
        model.add(LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W))
    model.add(Dropout(DROPOUT_AFTER_LSTM))
    model.add(Dense(1, activation='sigmoid'))
    metrics=['accuracy', 'fmeasure', 'precision', 'recall']
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model


model = create_model(embedding_weights_google, BIDIRECTIONAL)
print("Model created")

tensor_board = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=False)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint("models/%s.hdf5" % MODEL_NAME, monitor='val_acc', save_best_only=True, verbose=1)

## Train

model.fit(x_train, y_train,
                         nb_epoch=MAX_EPOCHES,
                         batch_size=BATCH_SIZE,
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=[tensor_board, early_stopping, model_checkpoint])

# serialize model to JSON
model_json = model.to_json()
with open(MODEL_NAME + ".json", "w") as json_file:
    json_file.write(model_json)
print("Model saved")
