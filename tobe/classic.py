import numpy as np

def create_mapping(column):
    ind2col = {}

    words = [row.split(' ') for row in column]
    unique = np.unqiue(np.array(words))


def get_x_y(df, embeddings):

    tag2ind = {key: i for i, key in enumerate([mask] + TO_BE_VARIANTS)}

    features = df.columns.values




    tags = list(df[df.columns[0]])
    texts = list(df[df.columns[1]])



def get_features(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
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
