"""Module for configuring FastAPI-app with CORS-support."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router


def create_app() -> FastAPI:
    """Create and configure FastAPI-app object with CORS-support.

    Returns:
        FastAPI: FastAPI-app object.
    """

    # TODO Change to env variable
    fastapi_app = FastAPI(root_path="/crossretrieval")
    # fastapi_app.mount(
    #     "/assets", StaticFiles(directory=config.DIR_STATIC), name="static"
    # )

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.include_router(router)

    return fastapi_app


app = create_app()
