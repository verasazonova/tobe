import spacy
import sys
import argparse
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np

from tobe.corpus import read_context_corpus, TO_BE_VARIANTS, mask, save_labeled_seq_corpus
import tobe.dl as dl

import logging

filename = 'resources/masked_text.txt'

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)


def train(num_epochs, filename, logs_filename):
    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    texts, tags = read_context_corpus(filename)

    print('Read in {} texts and {} tags'.format(len(texts), len(tags)))

    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    # for train_index, test_index in sss.split(texts, tags):
    #     train_texts, X = texts[train_index], texts[test_index]
    #     train_tags, y = tags[train_index], tags[test_index]

    train_texts, X, train_tags, y = train_test_split(texts, tags, test_size=0.2, stratify=tags)

    print('Read in {} texts and {} tags, {}, {}'.format(len(train_texts), len(train_tags), len(X), len(y)))
    test_texts, dev_texts, test_tags, dev_tags = train_test_split(X, y, test_size=0.5, stratify=y)

    print('Split into {} train {} dev and {} test sets'.format(len(train_texts), len(dev_texts), len(test_texts)))

    settings = {'nr_hidden': 100,
                'max_length': max(len(t.split()) for t in texts),
                'nr_class': len(tag2ind.keys()),
                'num_lstm': 2,
                'dropout': 0.5,
                'lr': 0.001,
                }

    print('Created tag index: {}'.format(tag2ind))
    print('Starting to train with settings: {}'.format(settings))

    dl.train(train_texts, train_tags, dev_texts, dev_tags, settings, tag2ind, batch_size=32, nb_epoch=num_epochs,
             logs_name=logs_filename)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('-n', '--num_epochs', help='Num epochs')
    parser.add_argument('--filename', help='Filename')
    parser.add_argument('--logs', help='Logs filename')
    parser.add_argument('--run', action='store_true', help='Train model')
    arguments = parser.parse_args()

    if arguments.train:
        train(int(arguments.num_epochs), arguments.filename, arguments.logs)

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