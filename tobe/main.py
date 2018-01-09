import sys
if sys.version_info < (3, 0):
    sys.stderr.write('This package is written for python3.  Please see README.md for installation')
    exit(1)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
from spacy.tokens import Doc
import logging

from sklearn.model_selection import train_test_split
import numpy as np

import spacy
import tobe.dl as dl
from tobe.corpus import read_context_corpus, TO_BE_VARIANTS, mask, Guttenberg

CONTEXT_LENGTH = 10
MODEL_PATH = 'models/weights.hdf5'


class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def get_features(docs, max_length):
    """Extract token ids from spacy doc

    Args:
        docs (list(spacy.Doc)): a list of texts analyzed with spacy
        max_length (int): the max length of the feature (2*context_len + 1)

    Returns:
        numpy.array(num_samples, num_tokens)
    """
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
    """Converts lists of tags to categorical arrays

    Args:
        tags (list(str)): a list of tags
        tag2ind dict(str, int): index for tags

    Returns:
        numpy.array(num_samples, num_classes)
    """
    ys = np.zeros((len(tags), len(tag2ind.keys())), dtype='int32')
    for i, tag in enumerate(tags):
        vector_id = tag2ind[tag]
        if vector_id >= 0:
            ys[i, vector_id] = 1
        else:
            ys[i, vector_id] = 0
    return ys


def featurize(texts, tags, tag2ind):
    """Converts texts, tags to numpy arrays for training and predicting

    Args:
        texts (list(str)): a list of texts of the same length (number of tokens)
        tags (list(str)): a list of single word tags
        tag2ind (dict(str, int)): tag index

    Returns:
        X (numpy.array(num_samples, num_tokens)): X data
        y (numpy.array(num_samples, num_classes)): y data
        embeddings (numpy.array(vocabulary_size, num_dimensions)): embeddings
    """

    logging.debug("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    logging.debug(nlp.pipeline)

    logging.debug("Parsing texts.")
    docs = list(nlp.pipe(texts))
    max_len = len(docs[0])
    X = get_features(docs, max_len)

    logging.debug('Got X: {}'.format(X.shape))

    y = get_cls_targets(tags, tag2ind)
    logging.debug('Got y: {}'.format(y.shape))

    embeddings = get_embeddings(nlp.vocab)

    return X, y, embeddings


def train_or_evaluate(num_epochs, filename, logs_filename, model_name, evaluate=False):
    """Train or evaluate a given model on a given file.

    Args:
        num_epochs (int): number of epochs to train for
        filename (str): path to the processed corpus
        logs_filename (str): path to the file to store logs
        model_name (str): path to the file to store the model or to read the model from
        evaluate (bool): evaluate or train

    Returns:

    """
    logging.info('Preparing the data from {}'.format(filename))
    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    if os.path.exists(filename):
        df = read_context_corpus(filename)
    else:
        logging.error('Input file not found in {}'.format(filename))
        exit(1)

    tags = list(df[df.columns[0]])
    texts = list(df[df.columns[1]])
    logging.info('Read in {} texts and {} tags'.format(len(texts), len(tags)))

    X, y, embeddings = featurize(texts, tags, tag2ind)

    train_X, X, train_y, y, _, dev_texts = train_test_split(X, y, texts, test_size=0.2, stratify=y, random_state=1)

    logging.debug('Read in {} texts and {} tags, {}, {}'.format(len(train_X), len(train_y), len(X), len(y)))
    test_X, dev_X, test_y, dev_y, _, dev_texts = train_test_split(X, y, dev_texts, test_size=0.5, stratify=y)

    logging.info('Split into {} train {} dev and {} test sets'.format(len(train_X), len(dev_X), len(test_X)))

    settings = {'nr_hidden': 100,
                'max_length': max(len(t.split()) for t in texts),
                'nr_class': len(tag2ind.keys()),
                'num_lstm': 2,
                'dropout': 0.5,
                'lr': 0.001,
                }

    logging.debug('Created tag index: {}'.format(tag2ind))

    if evaluate:
        logging.info('Reading the model')
        if os.path.exists(model_name):
            logging.info('Reading model from {}'.format(model_name))
            model = dl.read_model(model_name)
        elif os.path.exists(MODEL_PATH):
            logging.info('Reading default model from {}'.format(MODEL_PATH))
            model = dl.read_model(MODEL_PATH)
        else:
            logging.error('Model not found in {} nor in '.format(model_name, MODEL_PATH))
            exit(1)
        print('Evaluating on dev')
        dl.evaluate(model, (dev_X, dev_y), tag2ind, texts=dev_texts)

        print('Evaluating on test')
        dl.evaluate(model, (test_X, test_y), tag2ind)

    else:
        logging.info('Starting to train with settings: {}'.format(settings))
        os.makedirs(os.path.dirname(model_name), exist_ok=True)
        logging.info('Saving into {}'.format(model_name))
        dl.train(train_X, train_y, (dev_X, dev_y), (test_X, test_y),
                 settings, tag2ind, embeddings,
                 batch_size=32, nb_epoch=num_epochs,
                 logs_name=logs_filename, model_name=model_name)


def run(filenames, output_dir, context_len=CONTEXT_LENGTH):
    """Run the experiment defined in the exerice

    Args:
        filenames (list(str)): paths to the input files
        output_dir (str): path to the output dir (also dubbed to stdout)
        context_len (int): the context length of the model

    Returns:

    """
    keys = [mask] + TO_BE_VARIANTS
    tag2ind = {key: i for i, key in enumerate(keys)}

    logging.info('Loading the model')
    if os.path.exists(MODEL_PATH):
        logging.info('Reading default model from {}'.format(MODEL_PATH))
        model = dl.read_model(MODEL_PATH)
    else:
        logging.error('Model not found in {}'.format(MODEL_PATH))
        exit(1)

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as fin:
            n = int(fin.readline().strip())
            logging.debug('Expecting {} masked verbs'.format(n))
            text = [' '.join(fin.readlines())]
            corpus = Guttenberg(text, 1, context_len, with_pos=False, with_direct_speech=False, mask_only=True)

            tags = []
            texts = []
            for context, tag, feature in corpus:
                tags.append(tag)
                texts.append(context)

            logging.debug('Read in {} texts and {} tags'.format(len(texts), len(tags)))

            X, _, _ = featurize(texts, tags, tag2ind)

        pred = np.argmax(model.predict(X), axis=-1)
        with open(os.path.join(output_dir, 'result_{}'.format(filename)), 'w') as fout:
            for p in pred:
                print(keys[p])
                fout.write('{}\n'.format(keys[p]))
        print('-----')


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Train model')
    parser.add_argument('-n', '--num_epochs', default='50', help='Num epochs')
    parser.add_argument('--filenames', nargs='+', help='Filename')
    parser.add_argument('--logs', default='logs/log.csv', help='Logs filename')
    parser.add_argument('--model', default='models/train/weights.hdf5', help='Path to save the new model')
    arguments = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    if arguments.train or arguments.evaluate:
        if not arguments.filenames:
            filename = 'resources/processed_corpus_{}.txt'.format(CONTEXT_LENGTH)
        else:
            filename = arguments.filenames[0]
        train_or_evaluate(int(arguments.num_epochs), filename, arguments.logs, arguments.model, arguments.evaluate)

    else:
        if not arguments.filenames:
            print('Usage: main.py --filenames masked_text_1.txt masked_text_2.txt')
            print('       main.py -h ')
            print('for more options')
            exit()

        logging.info('The output will be printed to stdout and saved in the working directory.')
        logging.info('Output filename = results_[input_file_name]')
        run(arguments.filenames, '.')


if __name__ == '__main__':
    main()
