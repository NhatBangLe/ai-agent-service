from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.data.database import create_db_and_tables
from src.api.image import router as image_router
from src.api.label import router as label_router
from src.error import NotFoundError, InvalidArgumentError


# noinspection PyUnusedLocal
@asynccontextmanager
async def lifespan(api: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(router=image_router)
app.include_router(router=label_router)


# noinspection PyUnusedLocal
@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": exc.reason},
    )


# noinspection PyUnusedLocal
@app.exception_handler(InvalidArgumentError)
async def invalid_argument_exception_handler(request: Request, exc: InvalidArgumentError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": exc.reason},
    )
