import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

DEFAULT_SAVE_DIRECTORY = "/resource"


def _get_save_dir_path():
    return os.getenv("SAVE_FILE_DIRECTORY", DEFAULT_SAVE_DIRECTORY)


class IFileService(ABC):
    class FileMetadata(BaseModel):
        """
        Represents metadata information for a file.

        This class defines key attributes required for describing a file's metadata.
        It includes details such as the file's name, MIME type, and file path.

        :ivar name: Name of the file.
        :ivar mime_type: MIME type of the file.
        :ivar path: Path to the file.
        """
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
    async def get_file(self, file_id: str) -> FileMetadata | None:
        """
        Retrieves file information for a given file identifier. This method must be
        implemented by any subclass. It is designed to asynchronously fetch or
        retrieve details of a file.

        This is an abstract method and does not include implementation. It must be
        overridden in any derived class.

        :param file_id: A string representing the unique identifier of the file to be
            retrieved.
        :return: Either a FileMetadata object containing the file's details if the
            file is found, or None if no file is associated with the given identifier.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_file(self, file: SaveFile) -> FileMetadata:
        """
        Saves a file asynchronously.

        This method is intended to be implemented by subclasses for saving files
        with specific logic. It accepts a file object and returns an identifier
        or path for the saved file.

        :param file: The file data to be saved.
        :return: A string representing the identifier or path of the saved file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class LocalFileService(IFileService):
    PATH_SPLITTER = "_spt_"
    FILE_NAME_PATTERN = "{file_id}_spt_{file_name}"

    async def get_file(self, file_id):
        save_dir = Path(_get_save_dir_path())
        for path in save_dir.glob(f"{file_id}{self.PATH_SPLITTER}*"):
            file_name = path.name
            mime_type, _ = mimetypes.guess_type(file_name)
            return self.FileMetadata(name=file_name, mime_type=mime_type, path=str(path))
        return None

    async def save_file(self, file):
        image_id = str(uuid4())
        extension = mimetypes.guess_extension(file.mime_type)
        if extension is None:
            extension = ""
        save_name = self.FILE_NAME_PATTERN.format(file_id=image_id, file_name=f"{file.name}{extension}")
        save_path = os.path.join(_get_save_dir_path(), save_name)
        Path(save_path).write_bytes(file.data)

        return self.FileMetadata(name=save_name, mime_type=file.mime_type, path=save_path)
