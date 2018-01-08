import numpy as np
import argparse
import spacy


from tobe.corpus import read_context_corpus
from tobe.dl import WhitespaceTokenizer


def create_mapping(column):

    words = [word for row in column for word in row.split(' ')]
    print(np.array(words))
    unique = np.unique(np.asarray(words))
    return {key: i for i, key in enumerate(unique)}


def dataframe_to_arrays(df):

    mappings = {}
    for i, feature in enumerate(df.columns.values):
        if i != 1:
            mappings[i] = create_mapping((list(df[df.columns[i]])))

    texts = list(df[df.columns[1]])
    tags = list(df[df.columns[0]])

    print("Loading spaCy")
    nlp = spacy.load('en_vectors_web_lg')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    docs = nlp.pipe(texts)

    max_len = len(texts[0].split())
    print('Max len', max_len)

    Xs = get_features(docs, max_len)
    print('Got X', Xs.shape)

    ys = get_cls_targets(tags, max_len, mappings[0])
    print('Got y', ys.shape)

    feats = []
    for i, mapping in mappings.items():
        if i != 0:
            feats.append(get_cls_targets(list(df[df.columns[i]]), max_len, mapping))

    print('Got feats', [f.shape for f in feats])

    Xs = np.concatenate([Xs] + feats, axis=1)
    print(Xs.shape)


def get_features(docs, max_length):
    docs = list(docs)
    dim = docs[0][0].vector.shape[0]
    Xs = np.zeros((len(docs), max_length * dim), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            Xs[i, j:j+dim] = token.vector
            j += dim
    return Xs


def get_cls_targets(tags, max_length, tag2ind):
    dim = min(len(tags[0].split()), max_length)
    print(dim)
    print(len(tags))
    ys = np.zeros((len(tags), dim), dtype='int32')
    for i, tag in enumerate(tags):
        for j, t in enumerate(tag.split()):
            if j< dim:
                ys[i, j] = tag2ind[t]
    return ys


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--filename', help='Filename')
    arguments = parser.parse_args()

    df = read_context_corpus(arguments.filename)

    dataframe_to_arrays(df)

if __name__ == '__main__':
    main()