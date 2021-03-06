# coding: utf-8

from datetime import datetime
suffix = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))

MODEL_NAME = "lstm_word2vec_" + suffix
TRAIN_DATASETS = ["data/test_imdb.csv", "data/train_imdb.csv", "data/test_rt_en.csv", "data/train_rt_en.csv"]
TOKENIZER_NAME = "lstm_word2vec_tokenizer_" + suffix
WORD_TO_VEC_PATH = "google.gz"

RANDOM_SEED = 42

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100

LSTM_DIM = 128
BIDIRECTIONAL = False
EMBEDDING_DIM = 300
DROPOUT_U = 0.2
DROPOUT_W = 0.2
DROPOUT_BEFORE_LSTM = 0
DROPOUT_AFTER_LSTM = 0.2

MAX_EPOCHES = 100
BATCH_SIZE = 2048


import pickle

import pandas as pd
import numpy as np

np.random.seed(RANDOM_SEED)

from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from keras import initializations
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Layer
from keras.layers import Flatten, Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from gensim.models import KeyedVectors



## Load data

def load_data(data_files, test_size=0.1):
    datasets = []
    for dataset in data_files:
        datasets.append(pd.read_csv(dataset, sep="|"))
    whole_data = pd.concat(datasets)
    return train_test_split(whole_data, test_size=test_size, random_state=RANDOM_SEED)

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

word2vec_google = KeyedVectors.load_word2vec_format(WORD_TO_VEC_PATH, binary=True)
word2vec_google.init_sims(replace=True)
print("Loaded")

def get_embedding(word2vec_model, word):
    try:
        return word2vec_model.word_vec(word)
    except KeyError:
        return np.zeros(word2vec_model.syn0norm.shape[1])

    
embedding_weights_google = np.zeros((nb_words, word2vec_google.syn0norm.shape[1]))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_weights_google[i] = get_embedding(word2vec_google, word)


print("Embeddings matrix created")


## Create model

class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(Attention, self).__init__(** kwargs)

    def build(self, input_shape):
        print(input_shape)
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(Attention, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

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
        model.add(Bidirectional(LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W, return_sequences=True)))
    else:
        model.add(LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W, return_sequences=True))
    model.add(Dropout(DROPOUT_AFTER_LSTM))
    model.add(Attention())
    model.add(Dense(1, activation='sigmoid'))
    metrics=['accuracy', 'fmeasure', 'precision', 'recall']
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model


model = create_model(embedding_weights_google, BIDIRECTIONAL)
print("Model created")

tensor_board = TensorBoard(log_dir='./logs/logs_{}'.format(MODEL_NAME), histogram_freq=0, write_graph=False, write_images=False)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint("models/%s.hdf5" % MODEL_NAME, monitor='val_acc', save_best_only=True, verbose=0)
csv_logger = CSVLogger('training.log', append=False)

## Train

model.fit(x_train, y_train,
                         nb_epoch=MAX_EPOCHES,
                         batch_size=BATCH_SIZE,
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=[tensor_board, early_stopping, model_checkpoint, csv_logger])

# serialize model to JSON
model_json = model.to_json()
with open(MODEL_NAME + ".json", "w") as json_file:
    json_file.write(model_json)
print("Model saved")
