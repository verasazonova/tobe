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

    Xs = get_features(docs, max_len)

    ys = get_cls_targets(tags, mappings[0])

    feats = []
    for i, mapping in mappings.items():
        if i != 0:
            feats.append(get_cls_targets(list(df[df.columns[i]]), mapping))

    print(Xs.shape)
    print(ys.shape)
    print(f.shape for f in feats)



def get_features(docs, max_length):
    docs = list(docs)
    dim = docs[0][0].vector.shape[0]
    Xs = np.zeros((len(docs), max_length * dim), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            Xs[i, j:j+dim] = token.vector
    return Xs


def get_cls_targets(tags, tag2ind):
    ys = np.zeros((len(tags), len(tag2ind.keys())), dtype='int32')
    print(ys.shape)
    for i, tag in enumerate(tags):
        vector_id = tag2ind[tag]
        if vector_id >= 0:
            ys[i, vector_id] = 1
        else:
            ys[i, vector_id] = 0
    return ys


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--filename', help='Filename')
    arguments = parser.parse_args()

    df = read_context_corpus(arguments.filename)

    dataframe_to_arrays(df)

if __name__ == '__main__':
    main()