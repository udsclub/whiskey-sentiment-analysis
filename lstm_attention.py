# coding: utf-8

MODEL_NAME = "lstm_attn"
TRAIN_DATASETS = ["data/test_imdb.csv", "data/train_imdb.csv", "data/test_rt_en.csv", "data/train_rt_en.csv"]
WORD_TO_VEC_PATH = "GoogleNews-vectors-negative300.bin.gz"


RANDOM_SEED = 42

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 100

LSTM_DIM = 128
MASKING = False
BIDIRECTIONAL = False
EMBEDDING_DIM = 300
DROPOUT_U = 0.2
DROPOUT_W = 0.2
DROPOUT_BEFORE_LSTM = 0.2
DROPOUT_AFTER_LSTM = 0.2

MAX_EPOCHES = 100
BATCH_SIZE = 2048

import pickle
import os

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
from keras.models import Sequential, Model
from keras.layers import Dense, Layer, Input, merge, Lambda
from keras.layers import Flatten, Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger

from gensim.models import KeyedVectors

## Create folder for model

if not os.path.exists(MODEL_NAME):
    os.makedirs(MODEL_NAME)


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
    "shouldn't": "shouldn_`_t",
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

with open(MODEL_NAME + '/' + 'tokenizer', 'wb') as ofile:
    pickle.dump(tokenizer, ofile)
    ofile.close()

sequences_train = tokenizer.texts_to_sequences(train_data['prep_text'])
sequences_test = tokenizer.texts_to_sequences(test_data['prep_text'])

x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
y_train = train_data['label']
y_test = test_data['label']
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

class AttentionLayer(Layer):
    '''
    Attention layer.
    '''

    def __init__(self, init='glorot_uniform', **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.init = initializations.get(init)

    def build(self, input_shape):
        self.Uw = self.init((input_shape[-1],))
        self.b = self.init((input_shape[1],))
        self.trainable_weights = [self.Uw]
        super(AttentionLayer, self).build(input_shape)

    def compute_mask(self, input, mask):
        return mask

    def call(self, x, mask=None):
        multData = K.exp(K.tanh(K.dot(x, self.Uw) + self.b))
        if mask is not None:
            multData = mask * multData
        output = multData / (K.sum(multData, axis=1) + K.epsilon())[:, None]
        return K.reshape(output, (output.shape[0], output.shape[1], 1))

    def get_output_shape_for(self, input_shape):
        newShape = list(input_shape)
        newShape[-1] = 1
        return tuple(newShape)


def create_model(pretrained_embedding_weights=None):
    wordsInputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='words_input')
    if pretrained_embedding_weights is None:
        emb = Embedding(nb_words, EMBEDDING_DIM, mask_zero=MASKING)(wordsInputs)
    else:
        emb = Embedding(nb_words, EMBEDDING_DIM, mask_zero=MASKING, weights=[pretrained_embedding_weights],
                        trainable=False)(wordsInputs)
    if DROPOUT_BEFORE_LSTM != 0.0:
        emb = Dropout(DROPOUT_BEFORE_LSTM)(emb)
    if BIDIRECTIONAL:
        word_rnn = Bidirectional(LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W, return_sequences=True))(emb)
    else:
        word_rnn = LSTM(LSTM_DIM, dropout_U=DROPOUT_U, dropout_W=DROPOUT_W, return_sequences=True)(emb)
    if DROPOUT_AFTER_LSTM > 0.0:
        word_rnn = Dropout(DROPOUT_AFTER_LSTM)(word_rnn)
    attention = AttentionLayer()(word_rnn)
    sentence_emb = merge([word_rnn, attention], mode=lambda x: x[1] * x[0], output_shape=lambda x: x[0])
    sentence_emb = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda x: (x[0], x[2]))(sentence_emb)
    output = Dense(1, activation="sigmoid", name="documentOut")(sentence_emb)

    model = Model(input=[wordsInputs], output=[output])
    metrics = ['accuracy', 'fmeasure', 'precision', 'recall']
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
    return model


model = create_model(embedding_weights_google)
print("Model created")

# tensor_board = TensorBoard(log_dir='./logs/logs_{}'.format(MODEL_NAME), histogram_freq=0, write_graph=False, write_images=False)
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(MODEL_NAME + '/' + 'weights.hdf5', monitor='val_acc', save_best_only=True, verbose=0)
csv_logger = CSVLogger('training.log', append=False)

## Train

model.fit(x_train, y_train,
          nb_epoch=MAX_EPOCHES,
          batch_size=BATCH_SIZE,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping, model_checkpoint, csv_logger])

# serialize model to JSON
model_json = model.to_json()
with open(MODEL_NAME + '/' + 'structure.json', "w") as json_file:
    json_file.write(model_json)
print("Model saved")
