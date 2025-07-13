# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from pydantic import Field, StrictBytes, StrictStr
from typing import List, Optional, Tuple, Union
from typing_extensions import Annotated
from openapi_server.models.nearest_item import NearestItem


class BaseRetrievalApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseRetrievalApi.subclasses = BaseRetrievalApi.subclasses + (cls,)
    async def retrieval_image_post(
        self,
        topk: Optional[Annotated[int, Field(strict=False, ge=0)]],
        body: Optional[Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]]],
    ) -> List[NearestItem]:
        ...


    async def retrieval_text_get(
        self,
        text: StrictStr,
        topk: Optional[Annotated[int, Field(strict=False, ge=0)]],
    ) -> List[NearestItem]:
        ...
