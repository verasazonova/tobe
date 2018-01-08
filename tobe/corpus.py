from numpy.random import choice, sample
from collections import defaultdict
import os.path
import spacy
from polyglot.detect import Detector
from polyglot.detect.base import UnknownLanguage
import re
import argparse
import csv
import pandas as pd

TO_BE_VARIANTS = ['am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
mask = '----'


def preprocess(text):
    return re.sub('\s+', ' ', text.strip())


def mask_paragraph(paragraph_tokens, tomask):
    if isinstance(tomask, int):
        inds = [i for i, token in enumerate(paragraph_tokens) if token.lower_ in TO_BE_VARIANTS]
        inds_to_mask = choice(inds, tomask, replace=False)
    else:
        inds_to_mask = tomask
    masked = [mask if i in inds_to_mask else token.text for i, token in enumerate(paragraph_tokens)]

    return masked


class Guttenberg():
    def __init__(self, fin, masking_prob, context_len,
                 mask_all=False, with_pos=True, with_direct_speech=False, mask_only=False):
        if isinstance(fin, str):
            self.fin = open(fin, 'r', encoding='utf-8', errors='surrogateescape')
        else:
            self.fin = fin
        self.masking_prob = masking_prob
        self.tomask = TO_BE_VARIANTS
        self.mask = mask
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
        self.la_count = defaultdict(int)
        self.class_count = defaultdict(int)
        self.context_len = context_len
        self.mask_all = mask_all
        self.with_pos = with_pos
        self.max_len = 500
        self.with_direct_speech = with_direct_speech
        self.mask_only = mask_only
        print('Loaded')

    def __del__(self):
        if hasattr(self.fin, 'close'):
            self.fin.close()

    def to_context(self, paragraph):
        contexts = []
        features = []
        tags = []
        text = preprocess(paragraph)

        if self.mask_only:
            context_centers = [self.mask]
        else:
            context_centers = self.tomask + [self.mask]

        doc = self.nlp(text)
        masked_text = [self.mask if t.lower_ in self.tomask else t.lower_ for t in doc]

        for token in doc:
            tok = token.lower_
            if tok in context_centers:

                if not self.mask_all:
                    masked_text = [self.mask if t.i == token.i else t.lower_ for t in self.nlp(text)]

                self.class_count[tok] += 1
                tag = tok

                start = max(0, token.i - self.context_len)
                end = token.i + self.context_len + 1

                lefts = doc[start:token.i]
                rights = doc[token.i + 1:end]
                context = [t.lower_ for t in lefts] + [self.mask] + [t.lower_ for t in rights]

                padding = ['<eos>' for _ in range(self.context_len * 2 + 1 - len(context))]

                feat = []
                if self.with_pos:
                    masked_doc = self.nlp(' '.join(masked_text))
                    context_pos = [t.pos_ for t in masked_doc[start:end]] + padding
                    feat.append(' '.join(context_pos))

                if self.with_direct_speech:
                    feat.append(text.count('"') % 2)

                features.append(feat)
                contexts.append(' '.join(context + padding))
                tags.append(tag)

        return contexts, tags, features

    def __iter__(self):
        for line in self.fin:
            paragraph = line.strip()
            # Need to hack at large paragraphs oh well
            if len(paragraph.split(' ')) > self.max_len:
                print('Spliting a very long paragraph')
                tokens = paragraph.split(' ')
                chunks = [' '.join(tokens[i:i+self.max_len]) for i in range(int(len(tokens)/300))]
            else:
                chunks = [paragraph]
            for paragraph in chunks:
                contexts, tags, features = self.to_context(paragraph)
                for context, tag, feature in zip(contexts, tags, features):
                    yield context, tag, feature


def save_context_corpus(corpus, filename, num_lines=None):
    with open(os.path.join('resources', filename), 'w', encoding='utf-8') as fout:
        wr = csv.writer(fout, quoting=csv.QUOTE_ALL)
        for i, (context, tag, feature) in enumerate(corpus):
            wr.writerow([tag, context] + feature)
#            fout.write('{} {}\n'.format(tag, ' '.join(context)))
            if num_lines is not None and i == num_lines:
                break


def read_context_corpus(filename):
    df = pd.read_csv(filename, header=-1, dtype=str)
    return df


def save_english_by_paragraph(filename, filename_out):
    with open(filename) as fin, open(filename_out, 'w') as fout:
        paragraph = ''
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
                    if la == 'en':
                        fout.write('{}\n'.format(preprocess(paragraph)))
                        paragraph = ''


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--to_english', action='store_true', help='Extract english corpus only')
    parser.add_argument('--pos', action='store_true', help='With pos')
    parser.add_argument('--dir_speech', action='store_true', help='Direct speech')
    parser.add_argument('-n', dest='context_len', help='Context length')
    parser.add_argument('--output', default='processed_corpus', help='Output name')
    parser.add_argument('--filename', default='processed_corpus', help='Output name')
    arguments = parser.parse_args()

    if arguments.to_english:
        save_english_by_paragraph('resources/corpus.txt', arguments.output)
    else:
        corpus = Guttenberg(arguments.filename, 1, int(arguments.context_len),
                            with_pos=arguments.pos,
                            with_direct_speech=arguments.dir_speech)
        save_context_corpus(corpus, '{}_{}.txt'.format(arguments.output, arguments.context_len))
        print('Language distribution')
        print(sorted(corpus.la_count, key=lambda x: x[1], reverse=True))
        print('Classes distribution')
        print(sorted(corpus.class_count, key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    main()