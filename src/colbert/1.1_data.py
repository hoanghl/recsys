import argparse
import sys
from pathlib import Path

import polars as pl
import yaml
from loguru import logger
from transformers import AutoTokenizer

logger.remove()
logger.add(sys.stderr, level="INFO")


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path-config", type=str, default="colbert/configs.yaml", dest="path_conf"
    )
    parser.add_argument("--num-procs", "-n", type=int, default=4, dest="num_procs")

    args = parser.parse_args()

    return args


def main():
    # =========================================================
    # Parse args
    # =========================================================
    logger.info("Parse arguments")

    args = _parse_args()

    num_procs = args.num_procs

    # =========================================================
    # Load data and resources
    # =========================================================
    logger.info("Load data and resources")

    # Load config
    assert Path(args.path_conf).exists()
    with open(args.path_conf) as file:
        conf = yaml.safe_load(file)

    # Load corpus
    assert Path(conf["RAW_DATA"]["corpus"]).exists()
    corpus = pl.read_ndjson(conf["RAW_DATA"]["corpus"])

    # Load tokenizer
    assert Path(conf["PATH_TOKENIZER"]).exists()
    tokenizer = AutoTokenizer.from_pretrained(conf["PATH_TOKENIZER"])

    # =========================================================
    # Process corpus
    # =========================================================
    for i in range(num_procs):
        logger.info(f"Process documents: {i:02d}/{num_procs}")

        path_saved = Path(conf["PROCESSED"]["corpus"].replace("[i]", f"{i:02d}"))
        if path_saved.exists():
            continue

        n_doc_per_shard = int(len(corpus) / num_procs)
        ith_doc_start = i * n_doc_per_shard
        ith_doc_end = (i + 1) * n_doc_per_shard if i < num_procs - 1 else len(corpus)

        # Add special token
        tok_doc = conf["TOKEN"]["document"]

        corpus_shard = (
            corpus
            .select(
                pl.col("_id").alias("did"),
                pl.format(f"{tokenizer.cls_token} {tok_doc} {{}}", pl.col("text")).alias("text"),
                
            )
            [ith_doc_start : ith_doc_end]
        )  # fmt: skip

        tokenizer.pad_token = "[PAD]"
        corpus_tokenized = tokenizer(
            corpus_shard["text"].to_list(),
            add_special_tokens=False,
            truncation=True,
            padding=True,
            return_tensors="np",
        )

        corpus_shard = (
            corpus_shard
            .with_columns(
                pl.Series(corpus_tokenized['input_ids']).alias('tok_ids')
            )
        )  # fmt: skip

        # Save processed
        corpus_shard.write_parquet(path_saved)


if __name__ == "__main__":
    sys.exit(main())
