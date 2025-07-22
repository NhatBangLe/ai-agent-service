from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


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
        :ivar kwargs: Additional metadata.
        """
        id: str = Field(min_length=1, description="Unique identifier for the file.")
        name: str = Field(min_length=1, description="Name of the file.")
        mime_type: str | None = Field(default=None, min_length=1, description="MIME type of the file.")
        path: str = Field(min_length=1, description="Path to the file.")
        kwargs: dict[str, Any] | None = Field(default=None, description="Additional metadata.")

    class File(FileMetadata):
        data: bytes = Field(min_length=1, description="File content in bytes")

    class SaveFile(BaseModel):
        """
        Represents a file to be saved along with its metadata.

        This class is used to encapsulate the metadata and raw content
        of a file, including its name, MIME type, and byte data content.

        :ivar name: Name of the file.
        :ivar mime_type: MIME type of the file.
        :ivar data: File content in bytes.
        :ivar kwargs: Additional metadata.
        """
        name: str = Field(min_length=1, description="Name of the file.")
        mime_type: str | None = Field(default=None, min_length=1, description="MIME type of the file.")
        data: bytes = Field(min_length=1, description="File content in bytes")
        kwargs: dict[str, Any] | None = Field(default=None, description="Additional metadata.")

    @abstractmethod
    async def get_metadata_by_id(self, file_id: str) -> FileMetadata | None:
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
    async def get_file_by_id(self, file_id: str) -> File | None:
        """
        Retrieve a file by its unique identifier.

        This method is expected to fetch a file based on its provided unique identifier
        from a storage or system where files are maintained. The file object is returned
        if it exists; otherwise, None is returned. This method is asynchronous and must be
        implemented in any subclass that inherits it.

        :param file_id: A string representing the unique identifier of the file.
        :return: An instance of the File object if the file exists; otherwise, None.
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
