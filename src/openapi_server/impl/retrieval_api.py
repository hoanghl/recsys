from loguru import logger

from openapi_server.apis.retrieval_api_base import BaseRetrievalApi
from openapi_server.models.nearest_item import NearestItem


class RetrievalAPI(BaseRetrievalApi):
    async def retrieval_image_post(self, topk, body):
        return await super().retrieval_image_post(topk, body)

    async def retrieval_text_get(self, text, topk):
        logger.info("Inside herererere")

        return [
            NearestItem(item_id=1),
            NearestItem(item_id=2),
            NearestItem(item_id=3),
        ]
