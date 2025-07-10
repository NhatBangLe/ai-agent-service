import os
from typing import Annotated

from dependency_injector.wiring import Provide
from fastapi import Depends, Query

from .container import ApplicationContainer
from .service.document import IDocumentService
from .service.image import IImageService
from .service.label import ILabelService
from .util import SecureDownloadGenerator, PagingParams


def provide_download_generator():
    secret_key = os.getenv("DOWNLOAD_GENERATOR_SECRET_KEY", "your-super-secret-key-change-in-production")
    return SecureDownloadGenerator(secret_key)


DownloadGeneratorDepend = Annotated[SecureDownloadGenerator, Depends(provide_download_generator)]
PagingQuery = Annotated[PagingParams, Query()]

DocumentServiceDepend = Annotated[IDocumentService, Depends(
    Provide[ApplicationContainer.service_container.document_service])]
ImageServiceDepend = Annotated[IImageService, Depends(
    Provide[ApplicationContainer.service_container.image_service])]
LabelServiceDepend = Annotated[ILabelService, Depends(
    Provide[ApplicationContainer.service_container.label_service])]
