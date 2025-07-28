from dependency_injector import containers, providers

from .data.database import IDatabaseConnection
from .repository.interface.document import IDocumentRepository
from .repository.interface.file import IFileRepository
from .repository.interface.image import IImageRepository
from .repository.interface.label import ILabelRepository
from .repository.interface.thread import IThreadRepository
from .service.interface.agent import IAgentService
from .service.interface.document import IDocumentService
from .service.interface.export import IExportingService
from .service.interface.file import IFileService
from .service.interface.image import IImageService
from .service.interface.label import ILabelService
from .service.interface.thread import IThreadService


class ApplicationContainer(containers.DeclarativeContainer):
    db_connection = providers.Dependency(instance_of=IDatabaseConnection)

    image_repository = providers.Dependency(instance_of=IImageRepository)
    label_repository = providers.Dependency(instance_of=ILabelRepository)
    document_repository = providers.Dependency(instance_of=IDocumentRepository)
    thread_repository = providers.Dependency(instance_of=IThreadRepository)
    file_repository = providers.Dependency(instance_of=IFileRepository)

    file_service = providers.Dependency(instance_of=IFileService)
    image_service = providers.Dependency(instance_of=IImageService)
    document_service = providers.Dependency(instance_of=IDocumentService)
    label_service = providers.Dependency(instance_of=ILabelService)
    thread_service = providers.Dependency(instance_of=IThreadService)
    exporting_service = providers.Dependency(instance_of=IExportingService)

    agent_service = providers.Dependency(instance_of=IAgentService)