from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class IExportingService(ABC):
    class ExportedFileMetadata(BaseModel):
        name: str = Field(description="Name of the exported file.")
        path: str = Field(description="Path to the exported file.")
        mime_type: str = Field(description="MIME type of the exported file.")
        kwargs: dict[str, Any] | None = Field(default=None, description="Additional metadata.")

    @abstractmethod
    async def export_labeled_images_by_label_id(self, label_id: int) -> ExportedFileMetadata:
        """
        Exports labeled images associated with the provided label ID as a compressed file.
        This operation retrieves the label and associated images by their label ID.
        The resulting file is returned with its metadata.

        :param label_id: The unique identifier of the label whose associated images
            need to be exported.
        :return: Metadata of the exported file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def export_all_labeled_images(self) -> ExportedFileMetadata:
        """
        Exports all labeled images into a zip archive file.

        This asynchronous method fetches all labeled images through the image repository, organizes
        them into folders corresponding to their labels, and creates a compressed file for
        download. The compressed file is saved in the designated cache directory.

        :returns: Metadata for the exported file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
