from loguru import logger

from openapi_server.apis.resource_api_base import BaseResourceApi
from openapi_server.models.nearest_item import NearestItem


class ResourceAPI(BaseResourceApi):
    async def resource_text_get(self, text, topk):
        logger.info("Inside herererere")

        return [
            NearestItem(item_id=1),
            NearestItem(item_id=2),
            NearestItem(item_id=3),
        ]
