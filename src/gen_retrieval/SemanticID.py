from collections import defaultdict
from functools import partial
from multiprocessing import Manager, Queue

from loguru import logger
from numpy import ndarray
from sklearn.cluster import KMeans
from tqdm.contrib.concurrent import process_map

embeddings: ndarray = None


def _add_new_centroid(
    tree_centroids: dict,
    embd: ndarray | None,
    idx_parent: int | None = None,
) -> int:
    """Add new centroid to `tree_centroids`

    Args:
        tree_centroids (dict): shared dict storing the centroids
        embd (ndarray|None): embedding of centroid, resulted from KMeans
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


def _get_clusters(ids_corpus: list, num_clusters: int) -> list[dict]:
    """Use KMeans to find clusters

    Args:
        ids_corpus (list): corpus id of documents
        num_clusters (int): no. clusters

    Returns:
        list[dict]: each element contains the corpus ids of document and centroid of corresponding cluster
    """

    # TODO: HoangLe [Mar-16]: Handle case when len(ids_corpus) < num_clusters

    global embeddings

    map_2corpus = {i: idx for i, idx in enumerate(ids_corpus)}

    result = KMeans(num_clusters).fit(embeddings[ids_corpus])

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
    tree_centroids: dict,
    identifiers,
    num_clusters: int,
    pid: int,
):
    global embeddings

    while True:
        if tasks.empty():
            break

        ids_corpus, idx_parent = tasks.get(timeout=1)

        logger.debug(f"pid: {pid} - idx_parent: {idx_parent}")

        clusters = _get_clusters(ids_corpus, num_clusters)

        for i, cluster in enumerate(clusters):
            idx_new_centroid = _add_new_centroid(
                tree_centroids, cluster["centroid"], idx_parent
            )

            for idx_corpus in cluster["labels"]:
                identifiers[idx_corpus] += [i]

            if len(cluster["labels"]) <= num_clusters:
                for idx, label in enumerate(cluster["labels"]):
                    identifiers[label] += [idx]
            else:
                tasks.put((cluster["labels"], idx_new_centroid))


class SemanticID:
    def __init__(self, C: int):
        self.C = C

        self.tree_centroids: dict = None
        self.identifiers: dict = None
        self.embeddings: ndarray = None

    @classmethod
    def construct(
        cls, embeddings_inp: ndarray, C: int, num_procs: int = 5
    ) -> "SemanticID":
        """Construct the semantically hierarchical ID

        Args:
            embeddings_inp (ndarray): input embedding, has shape [N, d] where N is no. documents
            C (int): no. documents in each cluster
            num_procs (int, optional): no. processes in parallel. Defaults to 5.

        Returns:
            SemanticID: instance of class `SemanticID`
        """
        global embeddings
        embeddings = embeddings_inp

        n = embeddings_inp.shape[0]

        semantic_id = SemanticID(C)
        semantic_id.embeddings = embeddings_inp

        # Declare shared objects and initialize
        manager = Manager()

        tree_centroids = manager.dict()
        idx_root = _add_new_centroid(tree_centroids, None, None)

        tasks = manager.Queue()
        tasks.put_nowait((list(range(n)), idx_root))

        identifiers = manager.dict({i: [] for i in range(n)})

        # Trigger parallel processing
        _construct_core(tasks, tree_centroids, identifiers, C, 0)
        partial_consumer = partial(
            _construct_core, tasks, tree_centroids, identifiers, C
        )
        process_map(partial_consumer, range(num_procs), max_workers=num_procs)

        # Clone centroids and identifiers
        semantic_id.tree_centroids = tree_centroids.copy()
        semantic_id.identifiers = identifiers.copy()

        return semantic_id
