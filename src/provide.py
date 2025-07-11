from typing import Annotated

from dependency_injector.wiring import Provide

from .container import ApplicationContainer
from .data.database import IDatabaseConnection
from .repository.interface.document import IDocumentRepository
from .repository.interface.file import IFileRepository
from .repository.interface.image import IImageRepository
from .repository.interface.label import ILabelRepository
from .repository.interface.thread import IThreadRepository
from .service.interface.document import IDocumentService
from .service.interface.export import IExportingService
from .service.interface.file import IFileService
from .service.interface.image import IImageService
from .service.interface.label import ILabelService
from .service.interface.thread import IThreadService

DatabaseConnectionProvide = Annotated[IDatabaseConnection, Provide[ApplicationContainer.db_connection]]

FileRepositoryProvide = Annotated[IFileRepository, Provide[ApplicationContainer.file_repository]]
ImageRepositoryProvide = Annotated[IImageRepository, Provide[ApplicationContainer.image_repository]]
DocumentRepositoryProvide = Annotated[IDocumentRepository, Provide[ApplicationContainer.document_repository]]
LabelRepositoryProvide = Annotated[ILabelRepository, Provide[ApplicationContainer.label_repository]]
ThreadRepositoryProvide = Annotated[IThreadRepository, Provide[ApplicationContainer.thread_repository]]

FileServiceProvide = Annotated[IFileService, Provide[ApplicationContainer.file_service]]
ImageServiceProvide = Annotated[IImageService, Provide[ApplicationContainer.image_service]]
DocumentServiceProvide = Annotated[IDocumentService, Provide[ApplicationContainer.document_service]]
LabelServiceProvide = Annotated[ILabelService, Provide[ApplicationContainer.label_service]]
ThreadServiceProvide = Annotated[IThreadService, Provide[ApplicationContainer.thread_service]]
ExportingServiceProvide = Annotated[IExportingService, Provide[ApplicationContainer.exporting_service]]
