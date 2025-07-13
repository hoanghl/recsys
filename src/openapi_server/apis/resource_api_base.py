# coding: utf-8

from typing import ClassVar, Dict, List, Tuple  # noqa: F401

from pydantic import Field, StrictBytes, StrictStr
from typing import Any, Tuple, Union
from typing_extensions import Annotated
from fastapi.responses import FileResponse


class BaseResourceApi:
    subclasses: ClassVar[Tuple] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseResourceApi.subclasses = BaseResourceApi.subclasses + (cls,)
    async def resource_post(
        self,
        file: Union[StrictBytes, StrictStr, Tuple[StrictStr, StrictBytes]],
    ) -> object:
        ...


    async def resource_resource_id_get(
        self,
        resourceId: Annotated[StrictStr, Field(description="Resource ID")],
    ) -> FileResponse:
        ...
