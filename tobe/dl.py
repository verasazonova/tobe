"""
This example shows how to use an LSTM sentiment classification model trained using Keras in spaCy. spaCy splits the document into sentences, and each sentence is classified using the LSTM. The scores for the sentences are then aggregated to give the document score. This kind of hierarchical model is quite difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras example on this dataset performs quite poorly, because it cuts off the documents so that they're a fixed size. This hurts review accuracy a lot, because people often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""

import plac
import random
import pathlib
import cytoolz
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import thinc.extra.datasets
from spacy.compat import pickle
import spacy
from spacy.tokens import Doc

from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


class Metrics(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        print('{}, {}'.format(self.validation_data[0].shape, self.validation_data[1].shape))
        self.val_f1s = None
        self.val_precisions = None
        self.val_recalls = None

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        print(val_predict.shape)
        val_targ = np.argmax(self.validation_data[1], axis=-1)
        print(val_targ.shape)

        precision, recall, f_score, true_sum = precision_recall_fscore_support(val_targ.flatten(), val_predict.flatten())

        self.val_f1s.append(f_score)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print(' — val_f1: {} — val_precision: {}— val_recall {}'.format(f_score, precision, recall))
        return


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, np.asarray(labels, dtype='int32')


def get_features(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def get_targets(doc_tags, max_length, tag2ind):
    ys = np.zeros((len(doc_tags), max_length, len(tag2ind.keys())), dtype='int32')
    for i, doc in enumerate(doc_tags):
        j = 0
        for tag in doc:
            vector_id = tag2ind[tag]
            if vector_id >= 0:
                ys[i, j, vector_id] = 1
            else:
                ys[i, j, vector_id] = 0
            j += 1
            if j >= max_length:
                break
    return ys


def train(train_texts, train_tags, dev_texts, dev_tags,
          lstm_settings, tag2ind,
          batch_size=100,
          nb_epoch=5, by_sentence=True):

    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    print(nlp.pipeline)

    embeddings = get_embeddings(nlp.vocab)
    print(embeddings.shape)

    model = compile_lstm(embeddings, lstm_settings)

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    dev_docs = list(nlp.pipe(dev_texts))

    # if by_sentence:
    #     train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
    #     dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, lstm_settings['max_length'])
    dev_X = get_features(dev_docs, lstm_settings['max_length'])

    print('Got X: {}, {}'.format(train_X.shape, dev_X.shape))

    train_tags = get_targets(train_tags, lstm_settings['max_length'], tag2ind)
    dev_tags = get_targets(dev_tags, lstm_settings['max_length'], tag2ind)

    class_weights = {}

    print('Got y: {}, {}'.format(train_tags.shape, dev_tags.shape))

    metrics = Metrics((dev_X, dev_tags))

    model.fit(train_X, train_tags,
              validation_data=(dev_X, dev_tags),
              epochs=nb_epoch,
              callbacks=[metrics],
              batch_size=batch_size)

    return model


def compile_lstm(embeddings, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=100, #settings['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    for _ in range(settings['num_lstm']):
        model.add(Bidirectional(LSTM(settings['nr_hidden'],
                                     recurrent_dropout=settings['dropout'],
                                     return_sequences=True,
                                     dropout=settings['dropout'])))

    model.add(TimeDistributed(Dense(settings['nr_class'],
                                    activation='softmax')))

    model.compile(optimizer=Adam(lr=settings['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model_dir, texts, labels, max_length=100):
    # def create_pipeline(nlp):
    #     '''
    #     This could be a lambda, but named functions are easier to read in Python.
    #     '''
    #     return [nlp.tagger, nlp.parser, SentimentAnalyser.load(model_dir, nlp,
    #                                                            max_length=max_length)]
    #
    # nlp = spacy.load('en')
    # nlp.pipeline = create_pipeline(nlp)
    #
    # correct = 0
    # i = 0
    # for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
    #     correct += bool(doc.sentiment >= 0.5) == bool(labels[i])
    #     i += 1
    # return float(correct) / i
    pass




