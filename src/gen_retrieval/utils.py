import torch
from lightning import LightningModule
from loguru import logger
from polars import DataFrame
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration


# =================================================
# Data-associated components
# =================================================
class DSIDataset(Dataset):
    def __init__(self, conf: dict, corpus: DataFrame, queries: DataFrame, split: DataFrame, is_val: bool = False):
        super().__init__()

        self.split = split
        self.corpus = corpus
        self.queries = queries
        self.is_val = is_val

        if self.is_val:
            self.data_len = len(self.split)
        else:
            self.data_len = len(self.split) * conf["INDEXING_RETRIEVAL_RATIO"]

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index):
        if index >= len(self.split):
            # =================================================
            # Craft training/validation sample for indexing
            # =================================================

            # Get query and document
            row = self.corpus[index % len(self.corpus)].to_dicts()[0]
            seq_encode, attn_mask_encode = row["tok_ids_text"], row["attn_mask_text"]
            seq_decode, attn_mask_decode = row["tok_ids_semantic_id"], row["attn_mask_semantic_id"]

            # logger.debug(document)

            seq_encode_idx, seq_decode_idx = str(row["_id"]), str(row["_id"])
            task = "indexing"
        else:
            # =================================================
            # Craft training/validation sample for retrieval
            # =================================================

            # Get query and document
            row = self.split[index].to_dicts()[0]
            seq_encode, attn_mask_encode = row["tok_ids_text"], row["attn_mask_text"]
            seq_decode, attn_mask_decode = row["tok_ids_semantic_id"], row["attn_mask_semantic_id"]

            # logger.debug(document)

            seq_encode_idx, seq_decode_idx = row["query-id"], str(row["corpus-id"])
            task = "retrieval"

        return {
            "task": task,
            "seq_encode": torch.tensor(seq_encode, dtype=torch.long),
            "attn_mask_encode": torch.tensor(attn_mask_encode, dtype=torch.long),
            "seq_encode_idx": seq_encode_idx,
            "seq_decode": torch.tensor(seq_decode, dtype=torch.long),
            "attn_mask_decode": torch.tensor(attn_mask_decode, dtype=torch.long),
            "seq_decode_idx": seq_decode_idx,
        }


# =================================================
# Model-associated components
# =================================================


class LinearDecayWithWarmup(_LRScheduler):
    def __init__(self, optimizer, lr_base: float, num_steps_warmup: int, num_steps_max: int, last_epoch=-1):
        self.lr_base = lr_base
        self.num_steps_warmup = num_steps_warmup
        self.num_steps_max = num_steps_max

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.num_steps_warmup:
            return [self.lr_base * self._step_count / self.num_steps_warmup for _ in self.optimizer.param_groups]
        return [
            self.lr_base * (self._step_count - self.num_steps_warmup) / (self.num_steps_max - self.num_steps_warmup)
            for _ in self.optimizer.param_groups
        ]


class DSIModel(LightningModule):
    def __init__(self, conf: dict):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False

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

        self.log("loss_train", loss, on_step=True, on_epoch=True, batch_size=self.conf["BSZ"])

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

        sch = self.lr_schedulers()
        sch.step()

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
            out["lr_scheduler"] = LinearDecayWithWarmup(
                optimizer, float(self.conf["LR"]), self.conf["N_STEPS_WARMUP"], self.conf["MAX_STEPS"]
            )

        return out
