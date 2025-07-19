from src.apis.retrieval_api_base import BaseRetrievalApi
from src.models.nearest_item import NearestItem
from src.services import db
from src.services.embedding_extraction import EmbeddingExtraction


class RetrievalAPI(BaseRetrievalApi):
    def __init__(self):
        super().__init__()

        self.embd_extractor = EmbeddingExtraction()

    async def retrieval_image_post(self, topk, body):
        return await super().retrieval_image_post(topk, body)

    async def retrieval_text_get(self, text, topk):
        text_embd = self.embd_extractor.get_embd_text(text)[0].tolist()
        fetched = db.fetch_similar_items(text_embd=str(text_embd), topk=topk)

        ret = [NearestItem(item_id=entry["id"]) for entry in fetched]

        return ret
