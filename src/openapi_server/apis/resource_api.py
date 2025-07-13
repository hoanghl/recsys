# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from openapi_server.apis.resource_api_base import BaseResourceApi
import openapi_server.impl

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

from openapi_server.models.extra_models import TokenModel  # noqa: F401
from pydantic import Field, StrictBytes, StrictInt, StrictStr
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Annotated
from openapi_server.models.nearest_item import NearestItem


router = APIRouter()

ns_pkg = openapi_server.impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


@router.get(
    "/resource/image",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    tags=["Resource"],
    summary="Get top k similar resources via queried image",
    response_model_by_alias=True,
)
async def resource_image_get(
    topk: Optional[Annotated[int, Field(strict=True, ge=0)]] = Query(2, description="", alias="topk", ge=0),
    body: Optional[Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]]] = Body(None, description=""),
) -> List[NearestItem]:
    if not BaseResourceApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseResourceApi.subclasses[0]().resource_image_get(topk, body)


@router.post(
    "/resource",
    responses={
        200: {"model": object, "description": "Successful Response"},
        422: {"model": object, "description": "Validation Error"},
    },
    tags=["Resource"],
    summary="Upload item",
    response_model_by_alias=True,
)
async def resource_post(
    file: Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]] = Form(None, description=""),
) -> object:
    if not BaseResourceApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseResourceApi.subclasses[0]().resource_post(file)


@router.get(
    "/resource/text",
    responses={
        200: {"model": List[NearestItem], "description": "Successful Response"},
    },
    tags=["Resource"],
    summary="Get top k resources via query text",
    response_model_by_alias=True,
)
async def resource_text_get(
    text: StrictStr = Query(None, description="", alias="text"),
    topk: Optional[StrictInt] = Query(2, description="", alias="topk"),
) -> List[NearestItem]:
    if not BaseResourceApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseResourceApi.subclasses[0]().resource_text_get(text, topk)
