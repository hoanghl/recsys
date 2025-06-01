from typing import List

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .services import db
from .services.embedding_extraction import EmbeddingExtraction

router = APIRouter()
# templates = Jinja2Templates(directory="src/frontend/dist")

embd_extractor = EmbeddingExtraction()


# =================================================
# Validation
# =================================================
class ReturnGetNearestItems1(BaseModel):
    item_id: int


# =================================================
# Routes' definition
# =================================================


@router.get(
    "/api/resource",
    status_code=200,
    tags=["Resource"],
    summary="Get nearest items",
)
async def get_nearest_items(text: str, topk: int = 2) -> List[ReturnGetNearestItems1]:
    text_embd = embd_extractor.get_embd_text(text)[0].tolist()
    fetched = db.fetch_similar_items(text_embd=str(text_embd), topk=topk)

    ret = [{"item_id": entry["id"]} for entry in fetched]

    return ret


@router.put(
    "/api/resource",
    status_code=200,
    tags=["Resource"],
    summary="Upload item",
)
async def upload_item(file: UploadFile):
    # TODO: HoangLe [Jun-01]: Implement this
    pass
    # # Check the argument
    # file.filename

    # try:
    #     datetime.strptime(date, "%Y-%m-%d")
    # except ValueError:
    #     raise HTTPException(status_code=400, detail="Argument 'date' must be in format '2000-01-01'")

    # # Get DB
    # orders_df = db.fetch_orders(date=date)
    # order_daily_df = db.fetch_order_daily_snapshot(date=date)

    # if len(orders_df) == 0 or len(order_daily_df) == 0:
    #     raise HTTPException(status_code=400, detail=f"No data available on {date}")

    # # logger.debug(f"out1: {order_daily_df.to_dict(orient='records')}")
    # # logger.debug(f"out2: {orders_df.to_dict(orient='records')}")

    # ret = {
    #     "date": date,
    #     "total": order_daily_df.to_dict(orient="records"),
    #     "dishes": orders_df.to_dict(orient="records"),
    # }

    # return ret


@router.get(
    "/api/resource/{item_id}",
    status_code=200,
    tags=["Resource"],
    summary="Get specific item based on its id",
    response_class=FileResponse,
)
async def get_item(item_id: int):
    # Get database
    item_info = db.fetch_item(item_id=item_id)

    if len(item_info) == 0:
        raise HTTPException(status_code=400, detail=f"No item available for {item_id}")

    # Read file and return
    return FileResponse(path=item_info["location"])
