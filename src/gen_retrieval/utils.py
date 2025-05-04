import random
from typing import Iterator, Literal

import polars as pl
import torch
from lightning import LightningModule
from loguru import logger
from polars import DataFrame
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# =================================================
# Data-associated components
# =================================================
NEW_TOKENS_DICT = {"task_indexing_tok": "<IDX>", "task_retrieval_tok": "<RTRV>", "sep_tok": "<sep>"}


def _get_next_sentinel(tokenizer) -> Iterator[int]:
    for i in range(100):
        yield tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")


def _is_masked(mask_ratio: float = 0.15) -> bool:
    return random.random() <= mask_ratio


def _mask_seq(
    seq: list[int],
    tokenizer,
    mask_ratio: float = 0.15,
    span_length: int = 3,
) -> tuple[list, list]:
    seq_inp, seq_tgt = [], []
    gen_inp = _get_next_sentinel(tokenizer)
    gen_tgt = _get_next_sentinel(tokenizer)

    idx = 0
    is_sentinel_added = False
    while idx < len(seq):
        if _is_masked(mask_ratio):
            seq_inp.append(next(gen_inp))
            for _ in range(min(span_length, len(seq) - idx)):
                seq_tgt.append(seq[idx])
                idx += 1

            is_sentinel_added = False
        else:
            seq_inp.append(seq[idx])
            if not is_sentinel_added:
                is_sentinel_added = True
                seq_tgt.append(next(gen_tgt))

            idx += 1

    return seq_inp, seq_tgt


def _pad(
    seq: list[int],
    tokenizer,
    max_seq_length: int = 512,
) -> tuple[list, list]:
    len_pad = max_seq_length - len(seq)
    attn_mask = [1] * len(seq) + [0] * len_pad
    seq = seq + [tokenizer.pad_token_id] * len_pad

    return seq, attn_mask


def _build_sample(
    tokenizer,
    task: Literal["indexing", "retrieval"],
    special_toks: dict,
    inp: str,
    tgt: str | list[int],
    mask_ratio: float = 0.15,
    span_length: int = 3,
    max_seq_length: int = 512,
) -> tuple:
    """Build sample for both task: indexing and retrieval.
    If task is 'indexing', 'inp' and 'tgt' sequence are concatenated and masked.

    Args:
        tokenizer (_type_): tokenizer
        task (Literal[&#39;indexing&#39;, &#39;retrieval&#39;]): task to build sample
        special_toks (dict): special tokens
        inp (str): Input sequence
        tgt (str | list[int]): Target sequence
        mask_ratio (float, optional): Mask ratio. Only matters if task is 'indexing'. Defaults to .15.
        span_length (int, optional): Span length of masked phrase. Only matters if task is 'indexing'. Defaults to 3.
        max_seq_length (int, optional): Max sequence length. Defaults to 512.

    Raises:
        NotImplementedError: Raised when param 'task' is incorrect

    Returns:
        tuple: 4 sequences 'seq_encode', 'attn_mask_encode', 'seq_decode' and 'attn_mask_decode'
    """

    # logger.debug(f"Task: {task}")
    # logger.debug(f"inp : {inp}")
    # logger.debug(f"tgt : {tgt}")

    match task:
        case "indexing":
            # Token
            seq_inp: list[int] = tokenizer(inp, add_special_tokens=False)["input_ids"]
            if isinstance(tgt, list):
                seq_tgt = tokenizer.convert_tokens_to_ids(list(map(str, tgt)))
            else:
                seq_tgt: list[int] = tokenizer(tgt, add_special_tokens=False)["input_ids"]

            # Truncate
            max_len_inp = (
                max_seq_length - len(seq_tgt) - 1 - 1
            )  # one for task-specified token and another for separating token
            seq_inp = seq_inp[:max_len_inp]

            # Concat and add special tokens
            tok_id_sep = tokenizer.convert_tokens_to_ids(special_toks["sep_tok"])
            seq = seq_inp + [tok_id_sep] + seq_tgt

            # Mask
            seq_mask_inp, seq_mask_tgt = _mask_seq(seq, tokenizer, mask_ratio, span_length)

            # Add other special tokens
            tok_id_indexing = tokenizer.convert_tokens_to_ids(special_toks["task_indexing_tok"])
            seq_mask_inp = [tok_id_indexing] + seq_mask_inp
            seq_mask_tgt = seq_mask_tgt + [tokenizer.eos_token_id]

            # Pad and create mask

            # logger.debug(f"seq_mask_inp = {seq_mask_inp}")
            # logger.debug(f"seq_mask_tgt = {seq_mask_tgt}")

            seq_encode, attn_mask_encode = _pad(seq_mask_inp, tokenizer, max_seq_length)
            seq_decode, attn_mask_decode = _pad(seq_mask_tgt, tokenizer, max_seq_length)
        case "retrieval":
            # Token
            seq_inp: list[int] = tokenizer(inp, add_special_tokens=False)["input_ids"]
            if isinstance(tgt, list):
                seq_tgt = tokenizer.convert_tokens_to_ids(list(map(str, tgt)))
            else:
                seq_tgt: list[int] = tokenizer(tgt, add_special_tokens=False)["input_ids"]

            # Truncate
            max_len_inp = (
                max_seq_length - len(seq_tgt) - 1 - 1
            )  # one for task-specified token and another for separating token
            seq_inp = seq_inp[:max_len_inp]

            # Add special tokens
            tok_id_retrieval = tokenizer.convert_tokens_to_ids(special_toks["task_retrieval_tok"])
            seq_inp = [tok_id_retrieval] + seq_inp
            seq_tgt = seq_tgt + [tokenizer.eos_token_id]

            # Pad and create mask
            seq_encode, attn_mask_encode = _pad(seq_inp, tokenizer, max_seq_length)
            seq_decode, attn_mask_decode = _pad(seq_tgt, tokenizer, max_seq_length)
        case _:
            raise NotImplementedError()

    return seq_encode, attn_mask_encode, seq_decode, attn_mask_decode


def _is_index_task(ratio: float = 1.0 / 32) -> bool:
    return random.random() <= ratio


class DSIDataset(Dataset):
    def __init__(self, conf: dict, corpus: DataFrame, queries: DataFrame, split: DataFrame, is_val: bool = False):
        super().__init__()

        self.split = split
        self.corpus = corpus
        self.queries = queries
        self.is_val = is_val

        self.tokenizer = T5Tokenizer.from_pretrained(conf["MODEL_GENERATIVE"])
        self.tokenizer.add_tokens(list(NEW_TOKENS_DICT.values()))

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index):
        # Get query and document
        row = self.split[index].to_dicts()[0]
        query = self.queries.filter(pl.col("_id") == pl.lit(row["query-id"]))["text"].item()
        document = self.corpus.filter(pl.col("_id") == pl.lit(row["corpus-id"]))

        # logger.debug(document)

        doc_semantic_id = document["semantic_id"].item().to_list()
        doc_raw = document["text"].item()

        if not self.is_val:
            seq_encode, attn_mask_encode, seq_decode, attn_mask_decode = (
                _build_sample(self.tokenizer, "indexing", NEW_TOKENS_DICT, doc_raw, doc_semantic_id)
                if _is_index_task()
                else _build_sample(self.tokenizer, "retrieval", NEW_TOKENS_DICT, query, doc_semantic_id)
            )
        else:
            seq_encode, attn_mask_encode, seq_decode, attn_mask_decode = _build_sample(
                self.tokenizer, "retrieval", NEW_TOKENS_DICT, query, doc_semantic_id
            )

        return {
            "seq_encode": torch.tensor(seq_encode, dtype=torch.int32),
            "attn_mask_encode": torch.tensor(attn_mask_encode, dtype=torch.int32),
            "seq_decode": torch.tensor(seq_decode, dtype=torch.int32),
            "attn_mask_decode": torch.tensor(attn_mask_decode, dtype=torch.int32),
        }


# =================================================
# Model-associated components
# =================================================


class DSIModel(LightningModule):
    def __init__(self, conf: dict):
        super().__init__()

        self.save_hyperparameters()

        self.conf = conf

        self.model = T5ForConditionalGeneration.from_pretrained(conf["MODEL_GENERATIVE"])
        self.validations = []

    def training_step(self, batch, batch_id):
        # seq_encode, attn_mask_encode, seq_decode, attn_mask_decode = batch

        # logger.debug(f"seq_encode: {batch['seq_encode']}")
        # logger.debug(f"attn_mask_encode: {type(batch['attn_mask_encode'])}")
        # logger.debug(f"seq_decode: {type(seq_decode)}")
        # logger.debug(f"attn_mask_decode: {type(attn_mask_decode)}")

        loss = self.model(
            input_ids=batch["seq_encode"],
            attention_mask=batch["attn_mask_encode"],
            # decoder_input_ids=batch['seq_decode'],
            # decoder_attention_mask=batch['attn_mask_decode'],
            labels=batch["seq_decode"],
        ).loss

        self.log("loss_train", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_id):
        outputs = self.model.generate(batch["seq_encode"])

        logger.debug(f"outputs: {outputs}")

        self.validations.extend(
            [
                {"tgt": tgt, "pred": output}
                for tgt, output in zip(batch["seq_decode"].detach().cpu().tolist(), outputs.detach().cpu().tolist())
            ]
        )

    def on_validation_end(self):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.conf["LR"]))

        out = {"optimizer": optimizer}

        if self.conf["USE_LR_SCHEDULER"]:
            out["lr_scheduler"] = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.conf["NUM_EPOCHS"]
            )

        return out
