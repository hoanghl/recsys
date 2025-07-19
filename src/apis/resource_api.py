# coding: utf-8

from typing import Dict, List  # noqa: F401
import importlib
import pkgutil

from .resource_api_base import BaseResourceApi
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
from fastapi.responses import FileResponse

from pydantic import Field, StrictBytes, StrictStr
from typing import Tuple, Union
from typing_extensions import Annotated


router = APIRouter()

ns_pkg = impl
for _, name, _ in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + "."):
    importlib.import_module(name)


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
    "/resource/{resourceId}",
    responses={
        200: {"model": object, "description": "Successful Response"},
        422: {"model": object, "description": "Validation Error"},
    },
    tags=["Resource"],
    summary="Get resource via ID",
    response_model=None
)
async def resource_resource_id_get(
    resourceId: Annotated[StrictStr, Field(description="Resource ID")] = Path(..., description="Resource ID"),
) -> FileResponse:
    if not BaseResourceApi.subclasses:
        raise HTTPException(status_code=500, detail="Not implemented")
    return await BaseResourceApi.subclasses[0]().resource_resource_id_get(resourceId)
