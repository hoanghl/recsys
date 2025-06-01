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

DIR_STATIC = ""
match FASTAPI_ENV:
    case "development":
        DIR_STATIC = "src/frontend/dist/assets"
    case "production":
        DIR_STATIC = "/build/dist"
    case _:
        raise NotImplementedError("Unknown env configuration")


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
