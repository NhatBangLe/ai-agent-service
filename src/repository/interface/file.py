from abc import ABC
from uuid import UUID

from src.data.model import File
from src.repository import IRepository


class IFileRepository(IRepository[UUID, File], ABC):
    pass
