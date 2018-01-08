import argparse
import logging
import sys
from spacy.tokens import Doc
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import spacy
import tobe.dl as dl
from tobe.corpus import read_context_corpus, TO_BE_VARIANTS, mask, Guttenberg

CONTEXT_LENGTH = 20
MAX_LEN = 0

filename = 'resources/masked_text.txt'

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


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


def get_embeddings(vocab):
    return vocab.vectors.data


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


def featurize(texts, tags, tag2ind):
    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    print(nlp.pipeline)

    print("Parsing texts...")
    docs = list(nlp.pipe(texts))
    max_len = len(docs[0])
    X = get_features(docs, max_len)

    print('Got X: {}'.format(X.shape))

    y = get_cls_targets(tags, tag2ind)
    print('Got y: {}'.format(y.shape))

    return X, y


def train(num_epochs, filename, logs_filename, model_name, evaluate=False):
    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    df = read_context_corpus(filename)

    tags = list(df[df.columns[0]])
    texts = list(df[df.columns[1]])
    print('Read in {} texts and {} tags'.format(len(texts), len(tags)))

    X, y = featurize(texts, tags, tag2ind)

    train_X, X, train_y, y = train_test_split(X, y, test_size=0.2, stratify=y)

    print('Read in {} texts and {} tags, {}, {}'.format(len(train_X), len(train_y), len(X), len(y)))
    test_X, dev_X, test_y, dev_y = train_test_split(X, y, test_size=0.5, stratify=y)

    print('Split into {} train {} dev and {} test sets'.format(len(train_X), len(dev_X), len(test_X)))

    settings = {'nr_hidden': 100,
                # 'max_length': max(len(t.split()) for t in texts),
                'nr_class': len(tag2ind.keys()),
                'num_lstm': 2,
                'dropout': 0.5,
                'lr': 0.001,
                }

    print('Created tag index: {}'.format(tag2ind))
    print('Starting to train with settings: {}'.format(settings))

    if evaluate:
        print('reading the model')
        model = dl.read_model(model_name)
        print('Evaluating on dev')
        result = dl.evaluate(model, (dev_X, dev_y), tag2ind)

        print('Evaluating on test')
        result = dl.evaluate(model, (test_X, test_y), tag2ind)

    else:
        dl.train(train_X, train_y, (dev_X, dev_y), (test_X, test_y),
                 settings, tag2ind,
                 batch_size=32, nb_epoch=num_epochs,
                 logs_name=logs_filename, model_name=model_name)


def run(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        n = int(fin.readline().strip())
        corpus = Guttenberg(fin, 1, CONTEXT_LENGTH, with_pos=True, with_direct_speech=True)



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Train model')
    parser.add_argument('-n', '--num_epochs', default='0', help='Num epochs')
    parser.add_argument('--filename', help='Filename')
    parser.add_argument('--logs', help='Logs filename')
    parser.add_argument('--model', help='Model filename')
    parser.add_argument('--run', action='store_true', help='Train model')
    arguments = parser.parse_args()

    if arguments.train or arguments.evaluate:
        train(int(arguments.num_epochs), arguments.filename, arguments.logs, arguments.model, arguments.evaluate)

    elif arguments.run:
        print(sys.stdout.encoding)

        data = open(filename).readlines()
        if len(data) != 2:
            print('Error: {} lines in the file: {}'.format(len(data), filename))
        n = int(data[0].strip())
        paragraph = data[1].strip()

        print(n)
        print(paragraph)
        print()


if __name__ == '__main__':
    main()