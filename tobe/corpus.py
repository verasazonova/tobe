from numpy.random import choice, sample
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os.path
import spacy
from polyglot.detect import Detector
from polyglot.detect.base import UnknownLanguage
import re

TO_BE_VARIANTS = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
mask = '----'


def mask_paragraph(paragraph_tokens, tomask):
    if isinstance(tomask, int):
        inds = [i for i, token in enumerate(paragraph_tokens) if token.lower_ in TO_BE_VARIANTS]
        inds_to_mask = choice(inds, tomask, replace=False)
    else:
        inds_to_mask = tomask
    masked = [mask if i in inds_to_mask else token.text for i, token in enumerate(paragraph_tokens)]

    return masked


class Guttenberg():
    def __init__(self, filename, masking_prob):
        self.filename = filename
        self.masking_prob = masking_prob
        self.tomask = TO_BE_VARIANTS
        self.mask = mask
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.la_count = defaultdict(int)
        self.class_count = defaultdict(int)
        print('Loaded')

    def preprocess(self, paragraph):
        tokens = []
        tags = []
        doc = self.nlp(re.sub('\s+', ' ', paragraph.strip()))
        for token in doc:
            tok = token.lower_
            tag = 'O'
            if tok == self.mask:
                self.class_count[tok] += 1
            if tok in self.tomask:
                self.class_count[tok] += 1
                if sample() <= self.masking_prob:
                    tag = tok
                    tok = self.mask
            tokens.append(tok)
            tags.append(tag)
        return tokens, tags

    def __iter__(self):
        paragraph = ''
        with open(self.filename, 'r', encoding='utf-8', errors='surrogateescape') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    paragraph += ' ' + line
                else:
                    if paragraph:
                        try:
                            la = Detector(paragraph, quiet=True).language.code
                        except UnknownLanguage:
                            la = 'un'
                        self.la_count[la] += 1
                        if la == 'en':
                            data = self.preprocess(paragraph)
                            paragraph = ''
                            yield data


def save_preprocessed_corpus(corpus, filename):
    with open(os.path.join('resources', filename), 'w', encoding='utf-8') as fout:
        for tokens, tags in corpus:
            for token, tag in zip(tokens, tags):
                fout.write('{} {}\n'.format(token, tag))
            fout.write('\n')


def read_preprocessed_corpus(filename):
    texts = []
    tags = []
    paragraph = []
    paragraph_tags = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                data = line.strip().split()
                if len(data) != 2:
                    print('Error on line: {}'.format(line))
                    exit()
                token, tag = data
                paragraph.append(token)
                paragraph_tags.append(tag)
            else:
                texts.append(paragraph)
                tags.append(paragraph_tags)
                paragraph = []
                paragraph_tags = []
    return texts, tags


def contains_to_be(tokens):
    for token in tokens:
        if token in TO_BE_VARIANTS:
            return True

    return False


def count_classes(corpus, nlp):
    counts = defaultdict(int)
    for paragraph in corpus:
        doc = nlp(paragraph)
        for token in doc:
            key = token.lower_
            if key in TO_BE_VARIANTS:
                counts[key] += 1

    return counts


# TODO: should not read-in the whole file
def split_train_test_dev(corpus, train_per):
    processed_corpus = [data for data in corpus]

    train, test = train_test_split(processed_corpus, test_size=1-train_per)
    test, dev = train_test_split(test, test_size=0.5)

    for dataset, name in [ (train, 'train.txt'), (test, 'test.txt'), (dev, 'dev.txt')]:
        with open(os.path.join('resources',name), 'w', encoding='utf-8') as fout:
            for p in dataset:
                for token, tag in zip(p['tokens'], p['tags']):
                    fout.write('{} {}\n'.format(token, tag))
                fout.write('\n')


def main():
    corpus = Guttenberg('resources/corpus.txt', 0.6)
    save_preprocessed_corpus(corpus, 'preprocessed_corpus.txt')
    print('Language distribution')
    print(corpus.la_count)
    print('Classes distribution')
    print(corpus.class_count)

if __name__ == '__main__':
    main()