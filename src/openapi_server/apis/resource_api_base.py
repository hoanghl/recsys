# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from pydantic import Field, StrictBytes, StrictInt, StrictStr
from typing import Any, List, Optional, Tuple, Union
from typing_extensions import Annotated
from openapi_server.models.nearest_item import NearestItem


class BaseResourceApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseResourceApi.subclasses = BaseResourceApi.subclasses + (cls,)
    async def resource_image_get(
        self,
        topk: Optional[Annotated[int, Field(strict=True, ge=0)]],
        body: Optional[Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]]],
    ) -> List[NearestItem]:
        ...


    async def resource_post(
        self,
        file: Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]],
    ) -> object:
        ...


    async def resource_text_get(
        self,
        text: StrictStr,
        topk: Optional[StrictInt],
    ) -> List[NearestItem]:
        ...
