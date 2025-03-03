import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import argparse


import einops
import lightning as L
import numpy as np
import polars as pl
import torch
import yaml
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger
from polars import DataFrame
from torch import Tensor, nn
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel


path = "colbert/configs.yaml"
with open(path) as file:
    conf = yaml.safe_load(file)


version = datetime.now().strftime("%m-%d_%H-%M-%S")
project_name = "ColBERT"


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", dest="debug", action="store_true")
    parser.add_argument(
        "--mode",
        type=str,
        dest="mode",
        default="train",
        choices=["train", "val", "infer"],
    )
    parser.add_argument("--ckpt", type=str, dest="path_ckpt", default=None)

    args = parser.parse_args()

    return args


class Data(Dataset):
    def __init__(
        self,
        split: str,
        queries: DataFrame,
        corpus: DataFrame,
        punctuations: set,
        tok_id_mask: int,
        tok_id_pad: int,
        col_query_id: str = "qid",
        col_corpus_id: str = "did",
        col_tok_ids: str = "tok_ids",
    ):
        super().__init__()

        self.split = split
        self.punctuations = punctuations
        self.queries = queries
        self.col_query_id = col_query_id
        self.col_corpus_id = col_corpus_id
        self.col_tok_ids = col_tok_ids
        self.tok_id_mask = tok_id_mask
        self.tok_id_pad = tok_id_pad

        pairs = Data._load_pairs(split)

        # Reducing the size of corpus and positive pairs
        self.corpus = corpus.join(
            pairs.select(col_corpus_id).unique(), on=col_corpus_id, how="inner"
        )
        self.pairs = pairs.join(corpus, on=col_corpus_id, how="inner")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index):
        entry = self.pairs[index]
        qid, did = entry[self.col_query_id].item(), entry[self.col_corpus_id].item()

        tok_ids_query = (
            self.queries
            .filter(pl.col(self.col_query_id) == pl.lit(qid))[self.col_tok_ids]
            .item()
            .to_numpy()
            .copy()
        )  # fmt: skip
        tok_ids_doc = (
            self.corpus
            .filter(pl.col(self.col_corpus_id) == did)[self.col_tok_ids]
            .item()
            .to_numpy()
            .copy()
        )  # fmt: skip
        mask = torch.tensor(
            [
                tok.item() in self.punctuations or tok.item() == self.tok_id_pad
                for tok in tok_ids_doc
            ]
        )

        attention_mask_query = (tok_ids_query != self.tok_id_mask).astype(np.int32)
        attention_mask_doc = (tok_ids_doc != self.tok_id_pad).astype(np.int32)

        return {
            "query": tok_ids_query,
            "attention_mask_query": attention_mask_query,
            "attention_mask_doc": attention_mask_doc,
            "doc": tok_ids_doc,
            "mask": mask,
        }

    @staticmethod
    def _load_pairs(split: str) -> DataFrame:
        """Load positive pairs from train/val/tes split

        Args:
            split (str): split

        Returns:
            DataFrame: positive pairs
        """

        assert split in ["train", "val", "test"]
        assert Path(conf["RAW_DATA"][split]).exists()

        pairs = pl.read_csv(conf["RAW_DATA"]["train"], separator="\t").select(
            pl.col("query-id").alias("qid"), pl.col("corpus-id").alias("did")
        )

        return pairs


class ColBERT(Module):
    MAKS_VAL = -1e10

    def __init__(
        self,
        bert_model: str,
        size_vocab: int,
        d_hid: int = 128,
        d_hid_bert: int = 768,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(size_vocab)

        self.linear = nn.Linear(d_hid_bert, d_hid, bias=False)

    def forward(self, X: Tensor, attention_mask: Tensor) -> Tensor:
        # X: [bz, n]

        X = self.bert(X, attention_mask).last_hidden_state
        # [bz, n, d_hid_bert]

        X = self.linear(X)
        # [bz, n, d_hid]

        # X = X / X.norm(dim=-1, keepdim=True)
        X = torch.nn.functional.normalize(X, p=2, dim=-1)

        return X

    def trigger_train(
        self,
        query: Tensor,
        doc: Tensor,
        mask: Tensor,
        attention_mask_query: Tensor,
        attention_mask_doc: Tensor,
    ) -> Tensor:
        # query: [bz, Nd]
        # doc, mask: [bz, L]

        bz, Nd = query.shape

        ###################################################
        # Encode query and document
        ###################################################
        query = self.forward(query, attention_mask_query)
        # [bz, Nd, d_hid]
        doc = self.forward(doc, attention_mask_doc)
        # [bz, L, d_hid]

        ###################################################
        # Calculate the similarity
        ###################################################
        # Apply in-batch negative sampling
        query = einops.repeat(query, "b n d -> b repeat n d", repeat=bz)

        doc = einops.repeat(doc, "b l d -> repeat b l d", repeat=bz)
        doc = einops.rearrange(doc, "b a l d -> b a d l")

        sim = einops.einsum(query, doc, "b a n d, b a d l -> b a n l")
        # [bz, bz, Nd, L]

        # Mask positions which are the punctuation
        mask = einops.repeat(mask, "b l -> repeat1 b repeat2 l", repeat1=bz, repeat2=Nd)
        sim = sim.masked_fill(mask, ColBERT.MAKS_VAL)

        # Calculate score
        score = (sim.max(dim=-1).values).sum(dim=-1)
        # [bz, bz]

        logger.debug(f"score = {score}")

        # Calculate Listwise CE
        tgt = torch.arange(bz, dtype=torch.long, device=score.device)
        # [bz]

        loss = nn.functional.cross_entropy(score, tgt)

        return score, loss


class LitModel(L.LightningModule):
    def __init__(
        self,
        params: dict,
        lr: float = 1e-3,
        num_epochs: int = 10,
        use_lr_scheduler: bool = False,
        k: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.k = k
        self.num_epochs = num_epochs
        self.use_lr_scheduler = use_lr_scheduler

        self.model = ColBERT(**params)

    def forward(self, meal: Tensor) -> Any:
        return self.model(meal)

    def training_step(self, batch, batch_idx):
        _, loss = self.model.trigger_train(**batch)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        score, _ = self.model.trigger_train(**batch)
        # score: [bz, bz]

        # Calculate evaluation metrics
        bz = score.shape[0]
        relevances = torch.eye(bz, dtype=score.dtype, device=self.device)

        relevances_sorted = relevances[
            :, torch.argsort(score, dim=-1, descending=True)
        ][0, :, : self.k]

        val_ndcg = LitModel._calc_ndcg(relevances_sorted, self.k)
        val_mrr = LitModel._calc_mrr(relevances_sorted, self.k)
        val_map = LitModel._calc_map(relevances_sorted, self.k)

        self.log("val_ndcg", val_ndcg, on_epoch=True, prog_bar=False)
        self.log("val_mrr", val_mrr, on_epoch=True, prog_bar=False)
        self.log("val_map", val_map, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        out = {"optimizer": optimizer}

        if self.use_lr_scheduler:
            out["lr_scheduler"] = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=self.num_epochs,
            )

        return out

    @staticmethod
    def _calc_ndcg(relevances: Tensor, k: int) -> float:
        indices = 1 / torch.log2(
            torch.arange(2, relevances.shape[-1] + 2, device=relevances.device)
        ).unsqueeze(0)

        ndcg = torch.mean(relevances @ indices.T).item()

        return ndcg

    @staticmethod
    def _calc_mrr(relevances: Tensor, k: int) -> float:
        vals = 1 / ((relevances.argmax(dim=-1) + 1) * relevances.max(dim=-1).values)
        vals = vals.masked_fill(vals == torch.inf, 0)
        mrr = torch.mean(vals)

        return mrr

    @staticmethod
    def _calc_map(relevances: Tensor, k: int) -> float:
        val_map = (
            (
                relevances.cumsum(dim=-1)
                / (torch.arange(relevances.shape[-1], device=relevances.device) + 1)
            )
            .sum(-1)
            .mean()
        )
        return val_map


def train():
    ###################################################
    # Initial configurations
    ###################################################
    args = _parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")

    ###################################################
    # Load processed
    ###################################################

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf["PATHS"]["tokenizer"])
    tok_id_mask, tok_id_pad = tokenizer.convert_tokens_to_ids(["[MASK]", "[PAD]"])

    queries = pl.read_parquet(conf["PROCESSED"]["query"])

    path_raw = Path(conf["PROCESSED"]["corpus"].replace("[i]", "*"))
    paths = Path(path_raw.parent).glob(path_raw.stem)

    if args.debug:
        logger.debug("Load part of corpus")

        corpus = pl.read_parquet(list(paths)[0])
    else:
        logger.debug("Load full corpus")

        corpus = pl.concat([pl.read_parquet(path) for path in paths])

    corpus = corpus.with_columns(pl.col("did").cast(pl.Int64))

    # Load punctuations
    with open(conf["PATHS"]["punctuations"]) as file:
        map_punct2ids = json.load(file)

    punctuations = set(map_punct2ids.values())

    ###################################################
    # Train
    ###################################################
    # Declare data
    match args.mode:
        case "train":
            split = "train"
            shuffle = True
        case "val":
            split = "val"
            shuffle = False
        case _:
            raise NotImplementedError()

    data = Data(split, queries, corpus, punctuations, tok_id_mask, tok_id_pad)
    loader = DataLoader(
        data,
        batch_size=int(conf["BSZ"]),
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
    )

    # Declare model
    params = {
        "bert_model": conf["MODEL_NAME"],
        "size_vocab": len(tokenizer),
        "d_hid": conf["D_HID"],
        "d_hid_bert": conf["D_HID_BERT"],
    }
    litmodel = LitModel(params, lr=float(conf["LR"]), num_epochs=conf["NUM_EPOCHS"])

    # Declare trainer
    path_dir_ckpt = Path(conf["PATHS"]["ckpt_dir"])
    match conf["LOGGER"]:
        case "wandb":
            logger_pl = WandbLogger(
                name=project_name,
                save_dir=conf["PATHS"]["logs"],
                version=version,
                project=project_name,
            )
        case "tensorboard":
            logger_pl = TensorBoardLogger(
                conf["PATHS"]["logs"],
                name=project_name,
                version=version,
                default_hp_metric=False,
            )
        case _:
            raise NotImplementedError()

    trainer = L.Trainer(
        # devices=0,
        callbacks=[
            RichProgressBar(leave=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=path_dir_ckpt / project_name,
                filename=f"{path_dir_ckpt.stem}_{{epoch}}",
                every_n_epochs=2,
            ),
        ],
        logger=logger_pl,
        # gradient_clip_val=1,
        max_epochs=conf["NUM_EPOCHS"],
    )

    # Train or valiate
    if args.path_ckpt is not None:
        path_ckpt = Path(args.path_ckpt)
    else:
        path_ckpt = None

    match args.mode:
        case "train":
            trainer.fit(litmodel, loader, ckpt_path=path_ckpt)
        case "val":
            logger.debug(f"path_ckpt = {path_ckpt}")

            assert path_ckpt is not None and path_ckpt.exists()

            trainer.validate(litmodel, loader, ckpt_path=path_ckpt)
        case _:
            raise NotImplementedError()
