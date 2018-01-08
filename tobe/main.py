import argparse
import logging
import sys

from sklearn.model_selection import train_test_split

import tobe.dl as dl
from tobe.corpus import read_context_corpus, TO_BE_VARIANTS, mask, Guttenberg

CONTEXT_LENGTH = 20

filename = 'resources/masked_text.txt'

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(format=FORMAT)


def train(num_epochs, filename, logs_filename, model_name, evaluate=False):
    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    df = read_context_corpus(filename)

    tags = list(df[df.columns[0]])
    texts = list(df[df.columns[1]])
    print('Read in {} texts and {} tags'.format(len(texts), len(tags)))

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

    if evaluate:
        print('reading the model')
        model = dl.read_model(model_name)
        print('Evaluating')
        result = dl.evaluate(model, (test_texts, test_tags), tag2ind)
        print(result)

    else:
        dl.train(train_texts, train_tags, dev_texts, dev_tags, test_texts, test_tags,
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