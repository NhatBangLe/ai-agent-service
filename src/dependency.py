from typing import Annotated

from fastapi import Depends, Query
from sqlmodel import Session

from .data.database import get_session
from .util import SecureDownloadGenerator, PagingParams


def provide_download_generator():
    secret_key = "your-super-secret-key-change-in-production"
    return SecureDownloadGenerator(secret_key)


SessionDep = Annotated[Session, Depends(get_session)]
DownloadGeneratorDep = Annotated[SecureDownloadGenerator, Depends(provide_download_generator)]
PagingQuery = Annotated[PagingParams, Query()]
