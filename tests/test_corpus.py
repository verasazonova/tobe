from tobe.corpus import Guttenberg, save_context_corpus, read_context_corpus


def test_save_corpus():
    corpus = Guttenberg('resources/corpus.txt', 1, 4, with_pos=False, with_direct_speech=False)
    save_context_corpus(corpus, 'test_context_corpus.txt', 5)

    with open('resources/gold_context_corpus.txt') as fgold, open('resources/test_context_corpus.txt') as fin:
        gold = fgold.readlines()
        print(gold)
        processed = fin.readlines()
        print(processed)
        assert gold == processed


def test_save_corpus_with_features():
    corpus = Guttenberg('resources/corpus.txt', 1, 4, with_pos=True, with_direct_speech=True)
    save_context_corpus(corpus, 'test_context_corpus.txt', 5)

    df = read_context_corpus('resources/test_context_corpus.txt')

    assert len(df.columns) == 4
    assert len(df) == 6


def test_read_corpus():
    df = read_context_corpus('resources/gold_context_corpus.txt')
    gold_texts = ['this ebook ---- for the use of <eos> <eos>',
                  'xxxvii . the packet ---- opened 297 <eos> <eos>',
                  'white water . it ---- the first day of',
                  'seen . three canoes ---- engaged in the fascinating',
                  'in the nearest canoe ---- a fine figure of',
                  'on the bank had ---- holding his breath .']
    gold_tags = ['is', 'is', 'was', 'were', 'was', 'been']

    tags = list(df[df.columns[0]])
    texts = list(df[df.columns[1]])
    assert texts == gold_texts
    assert tags == gold_tags
