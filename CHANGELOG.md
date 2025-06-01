# CHANGELOG

# v1.0

- Use [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection) as primary text and image encoder
- Use `pgvector` to store image embedding
- Approximate nearest search is done using `L2 distance` in `pgvector`
- Input query is first input into CLIP to extract embedding, then search in postgreSQL to get
