from tobe.corpus import Guttenberg, save_context_corpus

def test_save_corpus():
    corpus = Guttenberg('resources/corpus.txt', 1)
    save_context_corpus(corpus, 'test_context_corpus.txt', 5)

    with open('resources/gold_context_corpus.txt') as fgold, open('resources/test_context_corpus.txt') as fin:
        gold = fgold.readlines()
        processed = fin.readlines()
        assert gold == processed
