from dependency_injector import containers, providers

from .file import IFileService, LocalFileService
from .image import IImageService, ImageServiceImpl
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository


class ServiceContainer(containers.DeclarativeContainer):
    image_repository = providers.Dependency(instance_of=IImageRepository)
    label_repository = providers.Dependency(instance_of=ILabelRepository)

    file_service = providers.Dependency(instance_of=IFileService, default=providers.Singleton(LocalFileService))
    image_service = providers.Dependency(instance_of=IImageService,
                                         default=providers.Singleton(ImageServiceImpl,
                                                                     image_repository=image_repository,
                                                                     label_repository=label_repository,
                                                                     file_service=file_service))
