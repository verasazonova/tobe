from numpy.random import choice

from collections import defaultdict

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
