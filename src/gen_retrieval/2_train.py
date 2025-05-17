import argparse
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
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
    cols = ["tok_ids", "attn_mask"]
    queries = (
        pl
        .read_parquet(conf['PROCESSED']['query'])
        .with_columns(
            *[
                pl.col(col).str.split(by='-').list.eval(pl.element().str.to_integer())
                for col in cols
            ]
        )
        .select('_id', 'tok_ids', 'attn_mask')
    )  # fmt: skip

    cols = ["tok_ids_text", "attn_mask_text", "tok_ids_semantic_id", "attn_mask_semantic_id"]
    corpus = (
        pl.read_parquet(conf['PROCESSED']['corpus'])
        .with_columns(
            *[
                pl.col(col).str.split(by='-').list.eval(pl.element().str.to_integer())
                for col in cols
            ]
        )
        .drop('text')
    )  # fmt: skip

    train = (
        pl.read_csv(conf['RAW_DATA']['train'], separator='\t')
        .join(queries, left_on='query-id', right_on='_id', how='left')
        .join(corpus, left_on='corpus-id', right_on='_id', how='left')
        .drop('score')
    )  # fmt: skip
    # val = (
    #     pl.read_csv(conf['RAW_DATA']['val'], separator='\t')
    #     .join(queries, left_on='query-id', right_on='_id', how='left')
    #     .join(corpus, left_on='corpus-id', right_on='_id', how='left')
    # ) # fmt: skip

    loader_train = DataLoader(DSIDataset(conf, corpus, queries, train), batch_size=conf["BSZ"], shuffle=True)

    # =================================================
    # Define model
    # =================================================
    version = datetime.now().strftime("%m-%d_%H-%M-%S")
    path_ckpt = Path(conf["PATHS"]["ckpt_dir"])
    match conf["LOGGER"]:
        case "wandb":
            logger = WandbLogger(
                name=version,
                save_dir=conf["PATHS"]["logs"],
                project=conf["PROJECT_NAME"],
            )
        case "tensorboard":
            logger = TensorBoardLogger(
                conf["PATHS"]["logs"],
                name=conf["PROJECT_NAME"],
                version=version,
                default_hp_metric=False,
            )

    lit_model = DSIModel(conf)
    trainer = Trainer(
        # precision=32,
        # accelerator="cpu",
        devices=2,
        log_every_n_steps=1,
        # num_sanity_val_steps=2,
        max_epochs=conf["MAX_EPOCHS"],
        # max_steps=conf["MAX_STEPS"],
        callbacks=[
            # RichProgressBar(leave=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=path_ckpt / conf["PROJECT_NAME"],
                filename=f"{path_ckpt.stem}_{{epoch}}",
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            ),
        ],
        logger=[logger],
        # check_val_every_n_epoch=4,
    )

    # =================================================
    # Train
    # =================================================
    trainer.fit(lit_model, loader_train, ckpt_path="ckpt/DSI/ckpt_epoch=0.ckpt")
    # trainer.validate(lit_model, loader_val)


if __name__ == "__main__":
    sys.exit(main())
