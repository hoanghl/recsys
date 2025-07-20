# coding: utf-8

from typing import List, Optional, Tuple, Union

from fastapi import APIRouter, Body, Query
from pydantic import Field, StrictBytes, StrictStr
from typing_extensions import Annotated

from src import services

from .model import NearestItem

router = APIRouter(tags=["Retrieval"], prefix="/retrieval")


@router.post(
    "/image",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    summary="Get top k similar resources via queried image",
    response_model_by_alias=True,
)
async def retrieval_image_post(
    topk: Optional[Annotated[int, Field(strict=True, ge=0)]] = Query(2, description="", alias="topk", ge=0),
    body: Optional[Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]]] = Body(None, description=""),
) -> List[NearestItem]:
    # TODO: HoangLe [Jul-20]: Implement this
    pass


@router.get(
    "/text",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    summary="Get top k resources via query text",
    response_model_by_alias=True,
)
async def retrieval_text_get(
    text: StrictStr = Query(None, description="", alias="text"),
    topk: Optional[Annotated[int, Field(strict=False, ge=0)]] = Query(2, description="", alias="topk", ge=0),
) -> List[NearestItem]:
    text_embd = services.embd_extractor.get_embd_text(text)[0].tolist()

    # TODO: HoangLe [Jul-20]: Fix this: Retrieve with embd_store instead
    # fetched = db.fetch_similar_items(text_embd=str(text_embd), topk=topk)
    fetched = [1, 2, 3]
    ret = [NearestItem(item_id=entry) for entry in fetched]

    return ret
