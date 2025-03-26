import argparse
import functools
import math
import sys
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from loguru import logger
from polars import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer


def _setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)


def _get_text(corpus: DataFrame):
    for row in corpus.iter_rows(named=True):
        yield row["text"]


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path-config",
        type=str,
        default="src/gen_retrieval/configs.yaml",
        dest="path_conf",
    )
    parser.add_argument("--num-procs", "-n", type=int, default=4, dest="num_procs")

    args = parser.parse_args()

    return args


def _f_tokenize(
    corpus: DataFrame,
    tokenizer,
    conf: dict,
    total_procs: int,
    proc_id: int,
):
    logger.info(f"Start process: {proc_id:02d}")

    # =================================================
    # Tokenize entire corpus
    # =================================================
    each = int(len(corpus) / total_procs)
    idx_start = proc_id * each
    idx_end = (proc_id + 1) * each if proc_id < total_procs - 1 else len(corpus)
    corpus = corpus[idx_start:idx_end]

    bsz = 200
    total = math.ceil(len(corpus) / bsz)

    input_ids = []
    # token_type_ids = []
    attention_mask = []

    idx = 0
    iterator = _get_text(corpus)

    if proc_id == 0:
        with tqdm(total=total) as pbar:
            while True:
                n = bsz if idx < total - 1 else len(corpus) - bsz * (total - 1)
                doc = [next(iterator) for _ in range(n)]

                encoded_input = tokenizer(
                    doc,
                    return_tensors="np",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                )
                input_ids.append(encoded_input["input_ids"])
                # token_type_ids.append(encoded_input['token_type_ids'])
                attention_mask.append(encoded_input["attention_mask"])
                # output = model(**encoded_input).last_hidden_state.mean(dim=1)
                # embds.append(output)

                pbar.update(1)
                idx += 1

                if idx == total:
                    break

    else:
        while True:
            n = bsz if idx < total - 1 else len(corpus) - bsz * (total - 1)
            doc = [next(iterator) for _ in range(n)]

            encoded_input = tokenizer(
                doc,
                return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            input_ids.append(encoded_input["input_ids"])
            # token_type_ids.append(encoded_input['token_type_ids'])
            attention_mask.append(encoded_input["attention_mask"])
            # output = model(**encoded_input).last_hidden_state.mean(dim=1)
            # embds.append(output)

            idx += 1

            if idx == total:
                break

    path = Path(
        conf["INTERIM"]["docid"]["tokens"].replace("[proc_id]", f"{proc_id:02d}")
    )
    path.parent.mkdir(exist_ok=True, parents=True)

    np.savez(
        path,
        {
            "input_ids": np.vstack(input_ids),
            "attention_mask": np.vstack(attention_mask),
        },
    )


def main():
    _setup_logger()

    # =================================================
    # Load things
    # =================================================
    # Load args
    args = _parse_args()

    # Load config
    with open(args.path_conf) as file:
        conf = yaml.safe_load(file)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf["MODEL_DOCID"])

    # Load corpus
    corpus = pl.read_ndjson(conf["RAW_DATA"]["corpus"])

    # =================================================
    # Tokenize
    # =================================================
    func = functools.partial(_f_tokenize, corpus, tokenizer, conf, args.num_procs)
    with Pool(args.num_procs) as pool:
        pool.map(func, range(args.num_procs))


if __name__ == "__main__":
    sys.exit(main())
