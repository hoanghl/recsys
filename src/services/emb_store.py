import sys
from typing import List

from loguru import logger
from numpy import ndarray
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src import config


class EmbdStoreUtils:
    def __init__(
        self,
        fieldname_id: str = "id",
        fieldname_embd: str = "embd",
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
    ):
        self.fieldname_id = fieldname_id
        self.fieldname_embd = fieldname_embd
        self.index_type = index_type
        self.metric_type = metric_type

        self.collection = self._initialize()

    def _initialize(self) -> Collection:
        # Connect to embedding store
        try:
            connections.connect("default", host=config.EMBDSTORE_HOST, port=config.EMBDSTORE_PORT)
            logger.info("Connected to Milvus.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus embedding store: {e}")

            sys.exit(1)

        # Create collection
        fields = [
            FieldSchema(name=self.fieldname_id, dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name=self.fieldname_embd, dtype=DataType.FLOAT16_VECTOR, dim=512),
        ]

        schema = CollectionSchema(fields)
        collection = Collection(config.EMBDSTORE_COLL_NAME, schema, consistency_level="Strong")
        return collection
    
    # def create_collection(self):

    def drop_collection(self):
        utility.drop_collection(config.EMBDSTORE_COLL_NAME)
        logger.info(f"Dropped collection '{config.EMBDSTORE_COLL_NAME}'.")

    def insert_data(self, entities: list):
        insert_result = self.collection.insert(entities)
        self.collection.flush()
        logger.info(f"Inserted data into '{self.collection.name}'. Number of entities: {self.collection.num_entities}")
        return insert_result

    def create_index(self):
        self.collection.create_index(
            self.fieldname_embd, {"index_type": self.index_type, "metric_type": self.metric_type}
        )

        logger.info(f"Index '{self.index_type}' created for field '{self.fieldname_embd}'.")

    def search(self, query: ndarray | List[ndarray], limit: int):
        if isinstance(query, ndarray):
            query = [query]

        self.collection.load()
        result = self.collection.search(
            query,
            self.fieldname_embd,
            {"metric_type": self.metric_type, "params": {"nprobe": 10}},
            limit=limit,
            output_fields=[self.fieldname_path],
        )

        return result
