from dependency_injector import containers, providers

from .document import IDocumentRepository, DocumentRepositoryImpl
from .file import IFileRepository, FileRepositoryImpl
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
    document_repository = providers.Dependency(
        instance_of=IDocumentRepository,
        default=providers.Singleton(DocumentRepositoryImpl, connection=db_connection))
    file_repository = providers.Dependency(
        instance_of=IFileRepository,
        default=providers.Singleton(FileRepositoryImpl, connection=db_connection))
