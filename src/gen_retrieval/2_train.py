import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from torch.utils.data import DataLoader

from .utils import DSIDataset, DSIModel


def _setup_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-config", type=str, default="src/gen_retrieval/configs.yaml", dest="path_conf")

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
    path = Path(args.path_conf)
    with open(path) as file:
        conf = yaml.safe_load(file)

    # =================================================
    # Define dataset
    # =================================================
    train = pl.read_csv(conf["RAW_DATA"]["train"], separator="\t")
    # val = pl.read_csv(conf["RAW_DATA"]["val"], separator="\t")
    queries = pl.read_ndjson(conf["RAW_DATA"]["query"])

    with open(conf["PROCESSED"]["semantic_docid"], "rb") as file:
        semantics_id = pickle.load(file)

    corpus = pl.read_ndjson(conf["RAW_DATA"]["corpus"]).with_columns(
        pl.col("_id").cast(pl.UInt32), pl.Series(semantics_id.values()).alias("semantic_id")
    )

    loader_train = DataLoader(DSIDataset(conf, corpus, queries, train), batch_size=conf["BSZ"])
    # loader_val = DataLoader(DSIDataset(corpus, queries, val, is_val=True), batch_size=conf["BSZ"])

    # =================================================
    # Define model
    # =================================================
    version = datetime.now().strftime("%m-%d_%H-%M-%S")
    path_ckpt = Path(conf["PATHS"]["ckpt_dir"])

    lit_model = DSIModel(conf)
    trainer = Trainer(
        # precision=32,
        # accelerator="cpu",
        devices=1,
        log_every_n_steps=1,
        num_sanity_val_steps=2,
        max_epochs=conf["NUM_EPOCHS"],
        callbacks=[
            # RichProgressBar(leave=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=path_ckpt / conf["PROJECT_NAME"], filename=f"{path_ckpt.stem}_{{epoch}}", every_n_epochs=1
            ),
        ],
        logger=[
            TensorBoardLogger(
                conf["PATHS"]["logs"],
                name=conf["PROJECT_NAME"],
                version=version,
                default_hp_metric=False,
            ),
            # WandbLogger(
            #         name=conf['PROJECT_NAME'],
            #         save_dir=conf["PATHS"]["logs"],
            #         version=version,
            #         project=conf['PROJECT_NAME'],
            #     )
        ],
        check_val_every_n_epoch=4,
    )

    # =================================================
    # Train
    # =================================================
    trainer.fit(lit_model, loader_train)
    # trainer.validate(lit_model, loader_val)


if __name__ == "__main__":
    sys.exit(main())
