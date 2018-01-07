from tobe.corpus import Guttenberg, save_context_corpus, read_context_corpus


def test_save_corpus():
    corpus = Guttenberg('resources/corpus.txt', 1, 4)
    save_context_corpus(corpus, 'test_context_corpus.txt', 5)

    with open('resources/gold_context_corpus.txt') as fgold, open('resources/test_context_corpus.txt') as fin:
        gold = fgold.readlines()
        processed = fin.readlines()
        assert gold == processed


def test_read_corpus():
    texts, tags = read_context_corpus('resources/gold_context_corpus.txt')
    gold_texts = ['---- for the use of <eos> <eos> <eos> <eos>',
                  'xxxvii . the packet ---- opened 297 <eos> <eos>',
                  'white water . it ---- the first day of',
                  'seen . three canoes ---- engaged in the fascinating',
                  'in the nearest canoe ---- a fine figure of',
                  'on the bank had ---- holding his breath .']
    gold_tags = ['is', 'is', 'was', 'were', 'was', 'been']

    assert texts == gold_texts
    assert tags == gold_tags
