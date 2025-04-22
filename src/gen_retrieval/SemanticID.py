import warnings
from collections import defaultdict
from functools import partial

import torch.multiprocessing as multiprocessing
from loguru import logger
from sklearn.cluster import KMeans
from torch import Tensor
from torch.multiprocessing import Manager, Queue


def _add_new_centroid(
    tree_centroids: dict,
    embd: Tensor | None,
    idx_parent: int | None = None,
) -> int:
    """Add new centroid to `tree_centroids`

    Args:
        tree_centroids (dict): shared dict storing the centroids
        embd (Tensor|None): embedding of centroid, resulted from KMeans
        idx_parent (int | None): index of parent node in `tree_centroids`

    Returns:
        int: Index in `tree_centroids` of newly added centroid
    """

    # Add new centroid to `tree_centroids`
    idx_new_centroid = len(tree_centroids)
    tree_centroids[idx_new_centroid] = {"embd": embd, "children": [], "documents": []}

    # Supplement new centroid to list of children of parent centroid
    if idx_parent is not None:
        assert idx_parent in tree_centroids
        tree_centroids[idx_parent]["children"].append(idx_new_centroid)


def _get_clusters(embeddings, ids_corpus: list, num_clusters: int) -> list[dict]:
    """Use KMeans to find clusters

    Args:
        ids_corpus (list): corpus id of documents
        num_clusters (int): no. clusters

    Returns:
        list[dict]: each element contains the corpus id of documents and centroid of corresponding cluster
    """

    if len(ids_corpus) <= num_clusters:
        out = [{"labels": [idx], "centroid": embeddings[idx]} for idx in ids_corpus]

    else:
        map_2corpus = {i: idx for i, idx in enumerate(ids_corpus)}

        pairs_try = [
            {"random_state": 0, "max_iter": 300},
            {"random_state": 37, "max_iter": 400},
            {"random_state": 42, "max_iter": 500},
        ]

        n_tries = 0
        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            for pair in pairs_try:
                try:
                    result = KMeans(num_clusters, random_state=pair["random_state"]).fit(embeddings[ids_corpus])
                    break
                except Warning:
                    n_tries += 1

        if n_tries == 3 and result.cluster_centers_.shape[0] == 1:
            # If after trying 3 times and no. clusters is 1, then assign each document into a separated cluster
            out = [{"labels": [idx], "centroid": embeddings[idx]} for idx in ids_corpus]
        else:
            clusters_labels = defaultdict(list)
            for i, label in enumerate(result.labels_):
                clusters_labels[label.item()].append(map_2corpus[i])

            out = [
                {"labels": clusters_labels[idx], "centroid": centroid}
                for idx, centroid in enumerate(result.cluster_centers_)
            ]

    return out


@logger.catch
def _construct_core(
    tasks: Queue,
    embeddings,
    tree_centroids: dict,
    identifiers,
    num_clusters: int,
    pid: int,
    is_init: bool = False,
):
    while True:
        if tasks.empty():
            break

        ids_corpus, idx_parent = tasks.get(timeout=1)

        logger.debug(f"pid: {pid} - Start")

        clusters = _get_clusters(embeddings, ids_corpus, num_clusters)

        for i, cluster in enumerate(clusters):
            idx_new_centroid = _add_new_centroid(tree_centroids, cluster["centroid"], idx_parent)

            for idx_corpus in cluster["labels"]:
                identifiers[idx_corpus] += [i]

            if len(cluster["labels"]) <= num_clusters:
                for idx, label in enumerate(cluster["labels"]):
                    identifiers[label] += [idx]
            else:
                tasks.put((cluster["labels"], idx_new_centroid))

        if is_init:
            break


class SemanticID:
    def __init__(self, C: int):
        self.C = C

        self.tree_centroids: dict = None
        self.identifiers: dict = None
        self.embeddings: Tensor = None

    @classmethod
    def construct(cls, embeddings_inp: Tensor, C: int, num_procs: int = 5) -> "SemanticID":
        """Construct the semantically hierarchical ID

        Args:
            embeddings_inp (Tensor): input embedding, has shape [N, d] where N is no. documents
            C (int): no. documents in each cluster
            num_procs (int, optional): no. processes in parallel. Defaults to 5.

        Returns:
            SemanticID: instance of class `SemanticID`
        """

        n = embeddings_inp.shape[0]

        semantic_id = SemanticID(C)
        semantic_id.embeddings = embeddings_inp
        semantic_id.embeddings.share_memory_()

        # Declare shared objects and initialize
        manager = Manager()

        tree_centroids = manager.dict()
        idx_root = _add_new_centroid(tree_centroids, None, None)

        tasks = manager.Queue()
        tasks.put_nowait((list(range(n)), idx_root))

        identifiers = manager.dict({i: [] for i in range(n)})

        # Trigger parallel processing
        _construct_core(tasks, embeddings_inp, tree_centroids, identifiers, C, 0, is_init=True)
        partial_consumer = partial(_construct_core, tasks, embeddings_inp, tree_centroids, identifiers, C)
        with multiprocessing.get_context("spawn").Pool(num_procs) as p:
            p.map(partial_consumer, range(num_procs))

        # Clone centroids and identifiers
        semantic_id.tree_centroids = tree_centroids
        semantic_id.identifiers = identifiers

        return semantic_id
