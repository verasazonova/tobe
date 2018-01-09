import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.optimizers import Adam
import pandas as pd
from collections import defaultdict

from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from keras.models import load_model

import logging


class Metrics(Callback):
    """A metrics class to calculate precision, recall and f1 score at the end of each epoch

    """
    def __init__(self, validation_data, tag2ind, logs_name):
        super().__init__()
        self.validation_data = validation_data
        self.tag2ind = tag2ind
        self.history = None
        self.logs_name = logs_name

    def on_train_begin(self, logs={}):
        self.history = defaultdict(list)

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        val_targ = np.argmax(self.validation_data[1], axis=-1)

        precision, recall, f_score, true_sum = precision_recall_fscore_support(val_targ, val_predict)
        for k, v in logs.items():
            self.history[k].append(v)
        for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
            for key, ind in self.tag2ind.items():
                self.history['{}: {}'.format(name, key)].append(m[ind])
            logging.info('\n{}: {}'.format(name, {key: m[ind] for key, ind in self.tag2ind.items()}))
        return

    def on_train_end(self, logs={}):
        pd.DataFrame.from_dict(self.history, orient='index').transpose().to_csv(self.logs_name)


def calculate_class_weight(y):
    train_y = np.argmax(y, axis=-1)
    class_weights = compute_class_weight('balanced', np.unique(train_y), train_y)
    logging.info('Calculated class_weights: {}'.format(class_weights))
    return class_weights


def train(train_X, train_y, dev_data, test_data,
          lstm_settings, tag2ind, embeddings,
          batch_size=100,
          nb_epoch=5, logs_name='logs.txt', model_name='models/weights.hdf5'):

    logging.info('Got embeddings with shape {}'.format(embeddings.shape))

    model = compile_lstm(embeddings, lstm_settings)

    metrics = Metrics(dev_data, tag2ind, logs_name)

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True,
                                   monitor='val_categorical_accuracy')

    model.fit(train_X, train_y,
              validation_data=dev_data,
              class_weight=calculate_class_weight(train_y),
              epochs=nb_epoch,
              callbacks=[metrics, checkpointer],
              batch_size=batch_size,
              shuffle=True)

    print('Evaluating on dev')
    evaluate(model, dev_data, tag2ind)

    print('Evaluating on test')
    evaluate(model, test_data, tag2ind)

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


def read_model(filename):
    return load_model(filename, compile=False)


def evaluate(model, test_data, tag2ind):
    val_predict = np.argmax(model.predict(test_data[0]), axis=-1)
    val_targ = np.argmax(test_data[1], axis=-1)

    ctx_len = (test_data[0].shape[1] - 1) / 2

    precision, recall, f_score, true_sum = precision_recall_fscore_support(val_targ, val_predict)
    print('all,{},{},{},{}'.format(ctx_len, precision, recall, f_score))

    result = {}
    for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
        for key, ind in tag2ind.items():
            result['{}: {}'.format(name, key)] = m[ind]

    for avg in ['micro', 'macro', 'weighted']:
        precision, recall, f_score, _ = precision_recall_fscore_support(val_targ, val_predict, average='micro')
        print('---,{},{},{},{},{}'.format(avg,ctx_len, precision, recall, f_score))

    for m, name in [(f_score, 'f1'), (precision, 'precision'), (recall, 'recall')]:
        result['OVERALL: {}'.format(name)] = m

    print(classification_report(val_targ, val_predict, target_names=tag2ind.keys()))

    print(tag2ind.keys())
    print(confusion_matrix(val_targ, val_predict))

    print(accuracy_score(val_targ, val_predict))


    return result
