import os
from typing import Annotated

from dependency_injector.wiring import Provide
from fastapi import Depends, Query

from .container import ApplicationContainer
from .service.interface.document import IDocumentService
from .service.interface.export import IExportingService
from .service.interface.file import IFileService
from .service.interface.image import IImageService
from .service.interface.label import ILabelService
from .service.interface.thread import IThreadService
from .util import SecureDownloadGenerator, PagingParams
from .util.constant import EnvVar


def provide_download_generator():
    secret_key = os.getenv(EnvVar.DOWNLOAD_GENERATOR_SECRET_KEY.value, "your-super-secret-key-change-in-production")
    return SecureDownloadGenerator(secret_key)


DownloadGeneratorDepend = Annotated[SecureDownloadGenerator, Depends(provide_download_generator)]
PagingQuery = Annotated[PagingParams, Query()]

FileServiceDepend = Annotated[IFileService, Depends(Provide[ApplicationContainer.file_service])]
DocumentServiceDepend = Annotated[IDocumentService, Depends(Provide[ApplicationContainer.document_service])]
ImageServiceDepend = Annotated[IImageService, Depends(Provide[ApplicationContainer.image_service])]
LabelServiceDepend = Annotated[ILabelService, Depends(Provide[ApplicationContainer.label_service])]
ExportingServiceDepend = Annotated[IExportingService, Depends(Provide[ApplicationContainer.exporting_service])]
ThreadServiceDepend = Annotated[IThreadService, Depends(Provide[ApplicationContainer.thread_service])]
