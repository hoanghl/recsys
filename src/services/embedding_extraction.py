import os
from io import BytesIO

import torch
from loguru import logger
from numpy import ndarray
from PIL import Image
from torch import Tensor
from transformers import CLIPProcessor, CLIPTokenizerFast

from src import config


class EmbeddingExtraction:
    def __init__(self):
        # Load model and other processing utilities
        logger.info("Load jit model and associated utilities")

        logger.debug(os.getcwd())

        self.loaded_model = torch.jit.load(config.PATH_MODEL, map_location="cpu").to(
            dtype=config.DTYPE, device=config.DEVICE
        )
        self.loaded_model.eval()

        self.processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(config.MODEL_NAME)

    def get_embd_text(self, text: str | list[str]) -> Tensor:
        token_ids = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        embd = (
            self.loaded_model.get_text_features(token_ids["input_ids"].to("mps"), token_ids["attention_mask"].to("mps"))
            .to("cpu")
            .detach()
        )

        return embd

    def get_embd_image(self, image_bytes: BytesIO) -> ndarray:
        """Extract embedding from input image

        Args:
            image_bytes (BytesIO): input image (already read into memory)

        Returns:
            ndarray: embedding vector
        """

        image = Image.open(image_bytes)
        out_sample_image = self.processor(images=image, return_tensors="pt")
        embd = (
            self.loaded_model.get_image_features(out_sample_image["pixel_values"].to("mps"))
            .to("cpu")
            .detach()
            .numpy()
            .squeeze()
        )

        return embd
