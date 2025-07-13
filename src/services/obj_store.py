from io import BytesIO

from loguru import logger
from minio import Minio

from src import config


class ObjStoreUtils:
    def __init__(self, bucket_name: str = "objects"):
        self.bucket_name = bucket_name

        self.client = Minio(
            endpoint=f"{config.OBJSTORE_HOST}:{config.OBJSTORE_PORT}",
            access_key=config.OBJSTORE_ROOT_USER,
            secret_key=config.OBJSTORE_ROOT_PWD,
            secure=False,
        )

        # Create bucket if not existed
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
            logger.info("Created bucket: ", bucket_name)

    def upload(self, obj_name: str, data: BytesIO, size: int):
        self.client.put_object(self.bucket_name, obj_name, data, size)

        logger.info(f"Put object '{obj_name}' to bucket '{self.bucket_name}'")

    def get(self, obj_name: str) -> bytes | None:
        response = None
        try:
            response = self.client.get_object(self.bucket_name, obj_name)
        finally:
            if response is not None:
                response.close()
                response.release_conn()

                return response.data
