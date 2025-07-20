from io import BytesIO
from pathlib import Path

from src import config


class ObjStoreUtils:
    def __init__(self):
        self.path_dir = Path(config.FILESTORE_DIR)

    def _get_path(self, filename: str) -> Path:
        name = Path(filename)
        path = self.path_dir / name.suffix[1:] / name

        path.parent.mkdir(exist_ok=True, parents=True)

        return path

    def save(self, filename: str, data: BytesIO):
        path = self._get_path(filename)

        with open(path, "wb+") as file:
            file.write(data)

    def load(self, filename: str) -> bytes:
        path = self._get_path(filename)

        with open(path, "rb") as file:
            return file.read()
