from typing import Annotated, Literal

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from sqlmodel import Session

from .data.database import get_session
from .util.main import SecureDownloadGenerator


def provide_download_generator():
    secret_key = "your-super-secret-key-change-in-production"
    return SecureDownloadGenerator(secret_key)


class PagingParams(BaseModel):
    offset: int = Field(description="The page number.", default=0, ge=0)
    limit: int = Field(description="The page size.", default=100, gt=0, le=100)
    order_by: Literal["created_at", "updated_at"] = "created_at"


SessionDep = Annotated[Session, Depends(get_session)]
DownloadGeneratorDep = Annotated[SecureDownloadGenerator, Depends(provide_download_generator)]
PagingQuery = Annotated[PagingParams, Query()]
