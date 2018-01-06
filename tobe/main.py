import spacy
import sys
from tobe.corpus import Guttenberg, count_classes


def main():
    print(sys.stdout.encoding)

    nlp = spacy.load('en')

    corpus = Guttenberg('corpus.txt')

    counts = count_classes(corpus, nlp)
    print(counts)


if __name__ == '__main__':
    main()