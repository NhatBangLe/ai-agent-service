from typing import Annotated

from fastapi import Depends
from sqlmodel import Session

from src.main import db_manager
from src.utility import SecureDownloadGenerator


def provide_download_generator():
    secret_key = "your-super-secret-key-change-in-production"
    return SecureDownloadGenerator(secret_key)


SessionDep = Annotated[Session, Depends(db_manager.get_session)]
DownloadGeneratorDep = Annotated[SecureDownloadGenerator, Depends(provide_download_generator)]