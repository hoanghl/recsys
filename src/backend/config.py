import os

import torch
from dotenv import load_dotenv

load_dotenv()

FASTAPI_ENV = os.getenv("FASTAPI_ENV", "development")

PORT = os.getenv("DB_PORT")
PWD = os.getenv("DB_PWD")
USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")

TABLE_ITEMS = os.getenv("TABLE_ITEMS", "items")


MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
PATH_MODEL: str = os.getenv("PATH_MODEL")
DEVICE: str = os.getenv("DEVICE", "cpu")
dtype = os.getenv("DTYPE", "bfloat16")
match dtype:
    case "float32":
        DTYPE = torch.float32
    case "bfloat16":
        DTYPE = torch.bfloat16
    case _:
        raise NotImplementedError()

# =================================================
# Configs for embedding store
# =================================================
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
