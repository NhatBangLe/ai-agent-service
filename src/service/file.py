import os
from pathlib import Path
from uuid import uuid4

from .interface.file import IFileService
from ..data.model import File
from ..repository.interface.file import IFileRepository
from ..util.constant import EnvVar
from ..util.function import strict_uuid_parser, shrink_file_name


class LocalFileService(IFileService):
    _file_repository: IFileRepository

    def __init__(self, file_repository: IFileRepository):
        super().__init__()
        self._file_repository = file_repository

    async def get_file_by_id(self, file_id):
        db_file = await self._file_repository.get_by_id(entity_id=strict_uuid_parser(file_id))
        if db_file is None:
            return None
        return self.FileMetadata(id=file_id, name=db_file.name,
                                 mime_type=db_file.mime_type,
                                 path=db_file.save_path)

    async def save_file(self, file):
        file_id = uuid4()
        save_path = Path(self.get_save_dir_path(), str(file_id))
        save_path.write_bytes(file.data)
        file_name = shrink_file_name(255, file.name)

        await self._file_repository.save(File(id=file_id,
                                              name=file_name,
                                              mime_type=file.mime_type,
                                              save_path=str(save_path)))
        return str(file_id)

    async def delete_file_by_id(self, file_id: str):
        deleted_file = await self._file_repository.delete_by_id(strict_uuid_parser(file_id))
        if deleted_file is None:
            return None
        os.remove(deleted_file.save_path)
        return self.FileMetadata(id=file_id, name=deleted_file.name,
                                 mime_type=deleted_file.mime_type, path=deleted_file.save_path)

    @staticmethod
    def get_save_dir_path():
        return os.getenv(EnvVar.SAVE_FILE_DIR.value, "/resource")
