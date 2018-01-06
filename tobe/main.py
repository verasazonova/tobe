from __future__ import unicode_literals, print_function
import spacy

from collections import defaultdict

import plac
import sys

TO_BE_VARIANTS = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']

def preprocess(paragraph):
    return paragraph.replace('\n', ' ')


class Guttenberg():
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        paragraph = ''
        with open(self.filename, 'r', encoding='utf-8', errors='surrogateescape') as fin:
            for line in fin:
                line = line.strip()
                if line:
                    paragraph += ' ' + line
                else:
                    str_to_yield = preprocess(paragraph)
                    paragraph = ''
                    yield str_to_yield


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


def main():
    print(sys.stdout.encoding)

    nlp = spacy.load('en')

    corpus = Guttenberg('corpus.txt')

    counts = count_classes(corpus, nlp)
    print(counts)


if __name__ == '__main__':
    main()