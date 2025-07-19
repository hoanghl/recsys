# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from .retrieval_api_base import BaseRetrievalApi
from src import impl

from fastapi import (  # noqa: F401
    APIRouter,
    Body,
    Cookie,
    Depends,
    Form,
    Header,
    HTTPException,
    Path,
    Query,
    Response,
    Security,
    status,
)
from loguru import logger

from pydantic import Field, StrictBytes, StrictStr
from typing import List, Optional, Tuple, Union
from typing_extensions import Annotated
from src.models.nearest_item import NearestItem


router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.post(
    "/retrieval/image",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    tags=["Retrieval"],
    summary="Get top k similar resources via queried image",
    response_model_by_alias=True,
)
async def retrieval_image_post(
    topk: Optional[Annotated[int, Field(strict=False, ge=0)]] = Query(2, description="", alias="topk", ge=0),
    body: Optional[Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]]] = Body(None, description=""),
) -> List[NearestItem]:
    if not BaseRetrievalApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseRetrievalApi.subclasses[0]().retrieval_image_post(topk, body)


@router.get(
    "/retrieval/text",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    tags=["Retrieval"],
    summary="Get top k resources via query text",
    response_model_by_alias=True,
)
async def retrieval_text_get(
    text: StrictStr = Query(None, description="", alias="text"),
    topk: Optional[Annotated[int, Field(strict=False, ge=0)]] = Query(2, description="", alias="topk", ge=0),
) -> List[NearestItem]:
    print(f"[HL] BaseRetrievalApi.subclasses: {BaseRetrievalApi.subclasses[0]}")

    out = await BaseRetrievalApi.subclasses[0]().retrieval_text_get(text, topk)

    logger.info(f"[HL] out: {out}")

    if not BaseRetrievalApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseRetrievalApi.subclasses[0]().retrieval_text_get(text, topk)
