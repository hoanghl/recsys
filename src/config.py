import os

import torch
from dotenv import load_dotenv

load_dotenv()

# =================================================
# Configs for web server
# =================================================
FASTAPI_ENV = os.getenv("FASTAPI_ENV", "development")
API_VERSION = os.getenv("VERSION")

PORT = os.getenv("DB_PORT")
PWD = os.getenv("DB_PWD")
USER = os.getenv("DB_USER")
DB_NAME = os.getenv("DB_NAME")
HOST = os.getenv("DB_HOST")

TABLE_ITEMS = os.getenv("TABLE_ITEMS", "items")


# =================================================
# Configs for embedding model
# =================================================
MODEL_NAME = os.getenv("MODEL_NAME", "openai/clip-vit-base-patch32")
PATH_MODEL: str = os.getenv("PATH_MODEL")
DEVICE: str = os.getenv("DEVICE", "cpu")
dtype = os.getenv("DTYPE", "float16")
match dtype:
    case "float32":
        DTYPE = torch.float32
    case "float16":
        DTYPE = torch.float16
    case _:
        raise NotImplementedError()


# =================================================
# Configs for embedding store
# =================================================
EMBDSTORE_COLL_NAME = os.getenv("EMBDSTORE_COLL_NAME")
EMBDSTORE_HOST = os.getenv("EMBDSTORE_HOST")
EMBDSTORE_PORT = os.getenv("EMBDSTORE_PORT")


# =================================================
# Configs for object store
# =================================================
FILESTORE_DIR = os.getenv("FILESTORE_DIR")
