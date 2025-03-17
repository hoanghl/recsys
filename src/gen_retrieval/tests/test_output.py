import numpy as np
from sklearn.cluster import KMeans

from src import SemanticID

SEED = 0


def test1():
    C = 10
    num_procs = 4
    n, d = 500, 64

    corpus_raw = np.random.random((n, d))

    corpus = corpus_raw.copy()

    identifier = []
    indices = np.arange(corpus.shape[0])
    while True:
        out = KMeans(C, random_state=0).fit(corpus)
        first = out.labels_[0]
        identifier.append(first.item())

        indices = np.argwhere(out.labels_ == first).flatten()
        corpus = corpus[indices]

        if len(indices) <= C:
            break

    identifier.append(0)

    semantic_id = SemanticID.construct(corpus_raw, C, num_procs)

    assert semantic_id.identifiers[0] == identifier
