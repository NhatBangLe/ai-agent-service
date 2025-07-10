from dependency_injector import containers, providers

from .document import IDocumentService, DocumentServiceImpl
from .file import IFileService, LocalFileService
from .image import IImageService, ImageServiceImpl
from .label import ILabelService, LabelServiceImpl
from ..repository.document import IDocumentRepository
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository


class ServiceContainer(containers.DeclarativeContainer):
    image_repository = providers.Dependency(instance_of=IImageRepository)
    label_repository = providers.Dependency(instance_of=ILabelRepository)
    document_repository = providers.Dependency(instance_of=IDocumentRepository)

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
