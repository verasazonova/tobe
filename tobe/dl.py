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
    ys = np.zeros((len(doc_tags), max_length), dtype='int32')
    for i, doc in enumerate(doc_tags):
        j = 0
        for tag in doc:
            vector_id = tag2ind[tag]
            if vector_id >= 0:
                ys[i, j] = vector_id
            else:
                ys[i, j] = 0
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

    print('Got y: {}, {}'.format(train_tags.shape, dev_tags.shape))

    model.fit(train_X, train_tags,
              validation_data=(dev_X, dev_tags),
              nb_epoch=nb_epoch,
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
                  metrics=['accuracy'])
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




