from dependency_injector import containers, providers

from .document import DocumentRepositoryImpl
from .interface.document import IDocumentRepository
from .file import FileRepositoryImpl
from .interface.file import IFileRepository
from .image import ImageRepositoryImpl
from .interface.image import IImageRepository
from .label import LabelRepositoryImpl
from .interface.label import ILabelRepository
from .thread import ThreadRepositoryImpl
from .interface.thread import IThreadRepository
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
    thread_repository = providers.Dependency(
        instance_of=IThreadRepository,
        default=providers.Singleton(ThreadRepositoryImpl, connection=db_connection))
