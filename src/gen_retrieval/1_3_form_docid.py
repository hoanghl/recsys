import argparse
import pickle
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

from .SemanticID import SemanticID


def _setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path-config",
        type=str,
        default="src/gen_retrieval/configs.yaml",
        dest="path_conf",
    )
    parser.add_argument(
        "--num-procs",
        "-n",
        type=int,
        default=4,
        dest="num_procs",
    )
    parser.add_argument(
        "--num-clusters",
        "-c",
        type=int,
        default=10,
        dest="num_clusters",
    )

    args = parser.parse_args()

    return args


def main():
    # =================================================
    # Load things
    # =================================================
    _setup_logger()

    # Load args
    args = _parse_args()

    # Load config
    path = Path("src/gen_retrieval/configs.yaml")
    with open(path) as file:
        conf = yaml.safe_load(file)

    path = Path(conf["INTERIM"]["docid"]["embds"])
    embeddings = torch.load(path)

    logger.info(f"embeddings: {embeddings.shape}")

    # =================================================
    # Form docID
    # =================================================
    semantic_ids = SemanticID.construct(
        embeddings, args.num_clusters, args.num_procs
    ).identifiers

    # Save embeddings
    logger.info(f"Save to {conf['INTERIM']['docid']['embds']}")

    path = Path(conf["PROCESSED"]["semantic_docid"])
    path.parent.mkdir(exist_ok=True, parents=True)

    with open(path, "wb+") as file:
        pickle.dump(semantic_ids, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug(semantic_ids)


if __name__ == "__main__":
    sys.exit(main())
