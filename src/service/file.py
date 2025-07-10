import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from dependency_injector.wiring import Provide
from pydantic import BaseModel, Field

from ..data.model import File
from ..repository.container import RepositoryContainer
from ..repository.file import IFileRepository
from ..util.function import strict_uuid_parser


class IFileService(ABC):
    class FileMetadata(BaseModel):
        """
        Represents metadata information for a file.

        This class defines key attributes required for describing a file's metadata.
        It includes details such as the file's name, MIME type, and file path.

        :ivar id: Unique identifier for the file.
        :ivar name: Name of the file.
        :ivar mime_type: MIME type of the file.
        :ivar path: Path to the file.
        """
        id: str = Field(min_length=1, description="Unique identifier for the file.")
        name: str = Field(min_length=1, max_length=100, description="Name of the file.")
        mime_type: str | None = Field(default=None, min_length=1, max_length=100,
                                      description="MIME type of the file.")
        path: str = Field(min_length=1, description="Path to the file.")

    class SaveFile(BaseModel):
        """
        Represents a file to be saved along with its metadata.

        This class is used to encapsulate the metadata and raw content
        of a file, including its name, MIME type, and byte data content.

        :ivar name: Name of the file.
        :ivar mime_type: MIME type of the file.
        :ivar data: File content in bytes.
        """
        name: str = Field(min_length=1, max_length=100, description="Name of the file.")
        mime_type: str | None = Field(default=None, min_length=1, max_length=100,
                                      description="MIME type of the file.")
        data: bytes = Field(min_length=1, description="File content in bytes")

    @abstractmethod
    async def get_file_by_id(self, file_id: str) -> FileMetadata | None:
        """
        Retrieve file metadata by file ID.

        This method fetches the metadata of a file stored.
        If the file does not exist, it returns None.
        Otherwise, it constructs and returns a metadata object.

        :param file_id: Unique identifier of the file to retrieve.
        :return: Metadata object containing file details or None if the file
                 does not exist.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_file(self, file: SaveFile) -> str:
        """
        Saves the provided file. Generates a unique identifier to ensure
        file uniqueness and constructs a save path dynamically.

        :param file: An object containing file data to be saved.
        :return: ID of the saved file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_file_by_id(self, file_id: str) -> FileMetadata | None:
        """
        Deletes a file record specified by file ID and removes the corresponding file.

        This function returns metadata of the deleted file if the operation is successful.
        If no file is found with the provided file ID, `None` is returned.

        :param file_id: Unique identifier of the file to be deleted.
        :return: Metadata of the deleted file if successful; otherwise, `None`.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class LocalFileService(IFileService):
    file_repository: Annotated[IFileRepository, Provide[RepositoryContainer.file_repository]]

    async def get_file_by_id(self, file_id):
        db_file = await self.file_repository.get_by_id(entity_id=strict_uuid_parser(file_id))
        if db_file is None:
            return None
        return self.FileMetadata(id=file_id, name=db_file.name,
                                 mime_type=db_file.mime_type,
                                 path=db_file.save_path)

    async def save_file(self, file):
        image_id = uuid4()
        save_path = Path(self.get_save_dir_path(), str(image_id))
        save_path.write_bytes(file.data)
        await self.file_repository.save(File(id=image_id,
                                             name=file.name,
                                             mime_type=file.mime_type,
                                             save_path=str(save_path)))
        return str(image_id)

    async def delete_file_by_id(self, file_id: str):
        deleted_file = await self.file_repository.delete_by_id(strict_uuid_parser(file_id))
        if deleted_file is None:
            return None
        os.remove(deleted_file.save_path)
        return self.FileMetadata(id=file_id, name=deleted_file.name,
                                 mime_type=deleted_file.mime_type, path=deleted_file.save_path)

    @staticmethod
    def get_save_dir_path():
        return os.getenv("SAVE_FILE_DIRECTORY", "/resource")
