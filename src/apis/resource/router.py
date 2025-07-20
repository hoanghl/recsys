# coding: utf-8


import io

from fastapi import APIRouter, Path, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import Field, StrictStr
from typing_extensions import Annotated

from src import services

router = APIRouter(
    prefix="/resource",
    tags=["Resource"],
)


@router.post(
    "",
    responses={
        200: {"model": object, "description": "Successful Response"},
        422: {"model": object, "description": "Validation Error"},
    },
    summary="Upload item",
    response_model_by_alias=True,
)
async def resource_post(file: UploadFile) -> object:
    # Encode resource
    raw = await file.read()
    embd = services.embd_extractor.get_embd_image(io.BytesIO(raw))

    logger.debug(f"embd: {type(embd)} - {embd}")

    # Store in Object store
    services.obj_store.save(file.filename, raw)

    # Store in Embedding store
    services.embd_store.insert_data([[embd]])


@router.get(
    "/{resourceId}",
    responses={
        200: {"model": object, "description": "Successful Response"},
        422: {"model": object, "description": "Validation Error"},
        404: {"model": object, "description": "Resource not found"},
    },
    summary="Get resource via ID",
    response_model=None,
)
async def resource_resource_id_get(
    resourceId: Annotated[StrictStr, Field(description="Resource ID")] = Path(..., description="Resource ID"),
) -> StreamingResponse:
    # TODO: HoangLe [Jul-20]: Implement this
    ...
