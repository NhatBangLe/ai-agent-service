from dependency_injector import containers, providers

from .document import DocumentServiceImpl
from .interface.document import IDocumentService
from .export import LocalExportingServiceImpl
from .interface.export import IExportingService
from .file import LocalFileService
from .interface.file import IFileService
from .image import ImageServiceImpl
from .interface.image import IImageService
from .label import LabelServiceImpl
from .interface.label import ILabelService
from .thread import ThreadServiceImpl
from .interface.thread import IThreadService
from ..repository.document import IDocumentRepository
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository
from ..repository.thread import IThreadRepository


class ServiceContainer(containers.DeclarativeContainer):
    image_repository = providers.Dependency(instance_of=IImageRepository)
    label_repository = providers.Dependency(instance_of=ILabelRepository)
    document_repository = providers.Dependency(instance_of=IDocumentRepository)
    thread_repository = providers.Dependency(instance_of=IThreadRepository)

    file_service = providers.Dependency(instance_of=IFileService, default=providers.Singleton(LocalFileService))
    image_service = providers.Dependency(instance_of=IImageService,
                                         default=providers.Singleton(ImageServiceImpl,
                                                                     image_repository=image_repository,
                                                                     label_repository=label_repository,
                                                                     file_service=file_service))
    document_service = providers.Dependency(instance_of=IDocumentService,
                                            default=providers.Singleton(DocumentServiceImpl,
                                                                        document_repository=document_repository,
                                                                        file_service=file_service))
    label_service = providers.Dependency(instance_of=ILabelService,
                                         default=providers.Singleton(LabelServiceImpl,
                                                                     label_repository=label_repository))
    thread_service = providers.Dependency(instance_of=IThreadService,
                                          default=providers.Singleton(ThreadServiceImpl,
                                                                      thread_repository=thread_repository))
    exporting_service = providers.Dependency(instance_of=IExportingService,
                                             default=providers.Singleton(LocalExportingServiceImpl,
                                                                         image_repository=image_repository,
                                                                         label_repository=label_repository))
