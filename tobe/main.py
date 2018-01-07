import spacy
import sys
import argparse
from sklearn.model_selection import train_test_split
import numpy as np

from tobe.corpus import read_context_corpus, TO_BE_VARIANTS, mask, save_labeled_seq_corpus
import tobe.dl as dl

import logging

filename = 'resources/masked_text.txt'

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)


def train():
    texts, tags = read_context_corpus('resources/preprocessed_corpus.txt')

    print('Read in {} texts and {} tags'.format(len(texts), len(tags)))

    train_texts, X, train_tags, y = train_test_split(texts, tags, test_size=0.2)

    print('Read in {} texts and {} tags, {}, {}'.format(len(train_texts), len(train_tags), len(X), len(y)))
    test_texts, dev_texts, test_tags, dev_tags = train_test_split(X, y, test_size=0.5)

    print('Split into {} train {} dev and {} test sets'.format(len(train_texts), len(dev_texts), len(test_texts)))

    # nr_hidden = 64
    # max_length = 100,  # Shape
    # dropout = 0.5, \
    #           learn_rate = 0.001,  # General NN config
    # nb_epoch = 5, batch_size = 100, nr_examples = -1

    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    settings = {'nr_hidden': 100,
                'max_length': max(len(t.split()) for t in texts),
                'nr_class': len(tag2ind.keys()),
                'num_lstm': 2,
                'dropout': 0.5,
                'lr': 0.001,
                }

    print('Created tag index: {}'.format(tag2ind))
    print('Starting to train with settings: {}'.format(settings))

    dl.train(train_texts, train_tags, dev_texts, dev_tags, settings, tag2ind, batch_size=32, nb_epoch=50)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--run', action='store_true', help='Train model')
    arguments = parser.parse_args()

    if arguments.train:
        train()

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