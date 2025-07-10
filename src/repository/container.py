from dependency_injector import containers, providers

from .image import IImageRepository, ImageRepositoryImpl
from .label import ILabelRepository, LabelRepositoryImpl
from ..data.database import IDatabaseConnection


class RepositoryContainer(containers.DeclarativeContainer):
    db_connection = providers.Dependency(instance_of=IDatabaseConnection)

    image_repository = providers.Dependency(
        instance_of=IImageRepository,
        default=providers.Singleton(ImageRepositoryImpl, connection=db_connection))
    label_repository = providers.Dependency(
        instance_of=ILabelRepository,
        default=providers.Singleton(LabelRepositoryImpl, connection=db_connection))
