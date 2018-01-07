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
        self.context_len = 4
        print('Loaded')

    def preprocess(self, text):
        return re.sub('\s+', ' ', text.strip())

    def to_labeled_seq(self, paragraph):
        tokens = []
        tags = []
        doc = self.nlp(self.preprocess(paragraph))
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

    def to_context(self, paragraph):
        contexts = []
        tags = []
        doc = self.nlp(self.preprocess(paragraph))
        for token in doc:
            tok = token.lower_
            if tok in self.tomask + [self.mask]:
                self.class_count[tok] += 1
                tag = tok
                lefts = doc[token.i - self.context_len:token.i]
                rights = doc[token.i + 1:token.i + self.context_len]
                context = [t.lower_ for t in lefts] + [self.mask] + [t.lower_ for t in rights]
                contexts.append(context + ['<eos>' for _ in range(self.context_len * 2 + 1 - len(context))])
                tags.append(tag)
        return contexts, tags

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
                            contexts, tags = self.to_context(paragraph)
                            paragraph = ''
                            for context, tag in zip(contexts, tags):
                                yield context, tag


def save_labeled_seq_corpus(corpus, filename):
    with open(os.path.join('resources', filename), 'w', encoding='utf-8') as fout:
        for tokens, tags in corpus:
            for token, tag in zip(tokens, tags):
                fout.write('{} {}\n'.format(token, tag))
            fout.write('\n')


def read_labeled_seq_corpus(filename):
    texts = []
    tags = []
    tokens = []
    token_tags = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line:
                data = line.strip().split()
                if len(data) != 2:
                    print('Error on line: {}'.format(line))
                    exit()
                token, tag = data
                tokens.append(token)
                token_tags.append(tag)
            else:
                texts.append(' '.join(tokens))
                tags.append(token_tags)
                tokens = []
                token_tags = []
    return texts, tags


def save_context_corpus(corpus, filename, num_lines=None):
    with open(os.path.join('resources', filename), 'w', encoding='utf-8') as fout:
        for i, (context, tag) in enumerate(corpus):
            fout.write('{} {}\n'.format(tag, ' '.join(context)))
            if num_lines is not None and i == num_lines:
                break


def read_context_forpus(filename):
    texts = []
    tags = []
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            data = line.strip().split()
            if len(data) != 2:
                print('Error on line: {}'.format(line))
                exit()
            tag, *tokens = data
            texts.append(' '.join(tokens))
            tags.append(tag)
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
    #save_labeled_seq_corpus(corpus, 'preprocessed_corpus.txt')
    save_context_corpus(corpus, 'preprocessed_corpus.txt')
    print('Language distribution')
    print(corpus.la_count)
    print('Classes distribution')
    print(corpus.class_count)

if __name__ == '__main__':
    main()