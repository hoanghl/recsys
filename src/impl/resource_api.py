from src.apis.resource_api_base import BaseResourceApi


class ResourceAPI(BaseResourceApi):
    async def resource_post(self, file):
        return await super().resource_post(file)

    async def resource_resource_id_get(self, resourceId):
        return await super().resource_resource_id_get(resourceId)
