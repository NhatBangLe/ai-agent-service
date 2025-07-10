from abc import ABC
from uuid import UUID

from . import IRepository, RepositoryImpl
from ..data.model import File


class IFileRepository(IRepository[UUID, File], ABC):
    pass


class FileRepositoryImpl(IFileRepository, RepositoryImpl):
    pass
