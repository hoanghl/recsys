import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel


def _setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)


class Embds(Dataset):
    def __init__(self, input_ids: Tensor, attention_mask: Tensor, device: str):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.device = device

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index].to(device=self.device),
            "attention_mask": self.attention_mask[index].to(device=self.device),
        }


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path-config",
        type=str,
        default="src/gen_retrieval/configs.yaml",
        dest="path_conf",
    )
    parser.add_argument(
        "--devices",
        "-d",
        type=str,
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        dest="device",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=128,
        dest="bsz",
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

    if (args.device == "cuda" and not torch.cuda.is_available()) or (
        args.device == "mps" and not torch.mps.is_available()
    ):
        logger.info(
            f"device set to '{args.device}' but '{args.device}' not available. Falling back to cpu."
        )

        args.device = "cpu"

    # Load model
    model = BertModel.from_pretrained(conf["MODEL_DOCID"]).to(device=args.device)
    model.eval()

    # Load corpus tokens
    input_ids_list, attention_mask_list = [], []
    paths = sorted(
        list(Path(conf["INTERIM"]["docid"]["tokens"]).parent.glob("corpus_tokens*.npz"))
    )
    for path in paths:
        logger.info(f"Load: {path.name}")

        loaded = np.load(path, allow_pickle=True)["arr_0"].item()

        input_ids_shard = torch.from_numpy(loaded["input_ids"])
        attention_mask_shard = torch.from_numpy(loaded["attention_mask"])

        input_ids_list.append(input_ids_shard)
        attention_mask_list.append(attention_mask_shard)

    input_ids = torch.vstack(input_ids_list)
    attention_mask = torch.vstack(attention_mask_list)

    logger.info(f"Load no. vectors: {len(input_ids)}")

    # =================================================
    # Extract embeddings
    # =================================================

    # Define dataloader
    dataset = Embds(input_ids, attention_mask, args.device)
    loader = DataLoader(dataset, batch_size=args.bsz)

    # Extract
    total = math.ceil(len(dataset) / args.bsz)

    embds_list = []
    with tqdm(total=total) as pbar, torch.no_grad():
        for batch in loader:
            out = model(**batch).last_hidden_state.mean(dim=1).cpu()
            embds_list.append(out)

            pbar.update(1)

    embds = torch.vstack(embds_list)

    # Save embeddings
    logger.info(f"Save to {conf['INTERIM']['docid']['embds']}")

    path = Path(conf["INTERIM"]["docid"]["embds"])
    path.parent.mkdir(exist_ok=True, parents=True)

    torch.save(embds, path)


if __name__ == "__main__":
    sys.exit(main())
