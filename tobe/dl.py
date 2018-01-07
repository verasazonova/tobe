import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.optimizers import Adam
import spacy
from spacy.tokens import Doc
import pandas as pd
from collections import defaultdict

from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight


class Metrics(Callback):
    def __init__(self, validation_data, tag2ind, logs_name):
        super().__init__()
        self.validation_data = validation_data
        self.tag2ind = tag2ind
        print('{}, {}'.format(self.validation_data[0].shape, self.validation_data[1].shape))
        self.history = None
        self.logs_name = logs_name

    def on_train_begin(self, logs={}):
        self.history = defaultdict(list)

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        val_targ = np.argmax(self.validation_data[1], axis=-1)

        precision, recall, f_score, true_sum = precision_recall_fscore_support(val_targ, val_predict) #flatten()
        for k, v in logs.items():
            self.history[k].append(v)
        for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
            for key, ind in self.tag2ind.items():
                self.history['{}: {}'.format(name, key)].append(m[ind])
            print('\n{}: {}'.format(name, {key: m[ind] for key, ind in self.tag2ind.items()}))
        return

    def on_train_end(self, logs={}):
        pd.DataFrame.from_dict(self.history, orient='index').transpose().to_csv(self.logs_name)


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


def get_cls_targets(tags, tag2ind):
    ys = np.zeros((len(tags), len(tag2ind.keys())), dtype='int32')
    print(ys.shape)
    for i, tag in enumerate(tags):
        vector_id = tag2ind[tag]
        if vector_id >= 0:
            ys[i, vector_id] = 1
        else:
            ys[i, vector_id] = 0
    return ys


def get_seq_targets(doc_tags,max_length, tag2ind):
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


def train(train_texts, train_tags, dev_texts, dev_tags, test_texts, test_tags,
          lstm_settings, tag2ind,
          batch_size=100,
          nb_epoch=5, by_sentence=True, logs_name='logs.txt', model_name='models/weights.hdf5'):

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

    train_tags = get_cls_targets(train_tags, tag2ind)
    dev_tags = get_cls_targets(dev_tags, tag2ind)

    print('Got y: {}, {}'.format(train_tags.shape, dev_tags.shape))

    train_y = np.argmax(train_tags, axis=-1)

    print(train_y.shape)
    print(np.unique(train_y))
    print(tag2ind.keys())

    class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
    print('Calculated class_weights: {}'.format(class_weights))

    metrics = Metrics((dev_X, dev_tags), tag2ind, logs_name)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)

    model.fit(train_X, train_tags,
              validation_data=(dev_X, dev_tags),
              class_weight=class_weights,
              epochs=nb_epoch,
              callbacks=[metrics, checkpointer],
              batch_size=batch_size,
              shuffle=True)

    evaluate(model, (test_texts, test_tags), tag2ind)

    return model


def compile_lstm(embeddings, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=settings['max_length'],
            trainable=False,
            weights=[embeddings],
            mask_zero=True
        )
    )
    for i in range(settings['num_lstm']):
        if i == settings['num_lstm'] - 1:
            sequence = False
        else:
            sequence = True
        model.add(Bidirectional(LSTM(settings['nr_hidden'],
                                     recurrent_dropout=settings['dropout'],
                                     return_sequences=sequence,
                                     dropout=settings['dropout'])))

    model.add(Dense(settings['nr_class'],
                    activation='softmax'))

    model.compile(optimizer=Adam(lr=settings['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


def evaluate(model, test_data, tag2ind):
    val_predict = np.argmax(model.predict(test_data[0]), axis=-1)
    val_targ = np.argmax(test_data[1], axis=-1)

    precision, recall, f_score, true_sum = precision_recall_fscore_support(val_targ, val_predict)

    result = {}
    for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
        for key, ind in tag2ind.items():
            result['{}: {}'.format(name, key)] = m[ind]
        print('\n{}: {}'.format(name, {key: m[ind] for key, ind in tag2ind.items()}))

    precision, recall, f_score, _ = precision_recall_fscore_support(val_targ, val_predict, average='micro')
    for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
        result['OVERALL: {}'.format(name)] = m
        print('\nOVERALL: {}'.format(name))

    return result
