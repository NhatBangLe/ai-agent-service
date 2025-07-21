from uuid import UUID

from . import RepositoryImpl
from .interface.file import IFileRepository
from ..data.model import File


class FileRepositoryImpl(IFileRepository, RepositoryImpl):

    async def get_by_id(self, entity_id: UUID) -> File | None:
        with self._connection.create_session() as session:
            entity = session.get(File, entity_id)
            return entity
