import datetime
from abc import ABC, abstractmethod
from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide
from fastapi import UploadFile

from .container import ServiceContainer
from .file import IFileService
from ..data.model import Image
from ..repository.container import RepositoryContainer
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError


class IImageService(ABC):

    @abstractmethod
    async def get_image_by_id(self, image_id: UUID) -> Image:
        """
        Retrieves an image by its unique identifier.

        This abstract method is intended to be implemented by a subclass. It is used
        to fetch an image using its unique identifier. The method is an asynchronous
        operation that must be awaited.

        :param image_id: The unique identifier of the image to be retrieved.
        :type image_id: UUID
        :return: The image corresponding to the provided identifier.
        :rtype: Image
        :raises NotImplementedError: If the method is not implemented in the subclass.
        :raises NotFoundError: If the image with the specified ID does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        """
        Saves an image uploaded by the user asynchronously. The method associates the
        uploaded file with the provided user ID and returns the unique identifier for the
        stored image. It is meant to be implemented by subclasses and raises a
        NotImplementedError if called directly from the base class.

        :param user_id: The unique identifier of the user to whom the image belongs.
        :type user_id: UUID
        :param file: The file object representing the image to be saved.
        :type file: UploadFile
        :return: A unique identifier for the saved image.
        :rtype: UUID
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        """
        Assigns a list of labels to a specific image.

        Assigns the given labels, identified by their IDs, to the specified image
        using its unique identifier. This is an abstract method and must be
        implemented in a subclass.

        :param image_id: The unique identifier for the image to assign labels to.
        :type image_id: UUID
        :param label_ids: A list of unique identifiers for the labels to be assigned
            to the image.
        :type label_ids: list[int]
        :raises NotImplementedError: If the method is not implemented.
        :raises NotFoundError: If the image with the specified ID does not exist.
        """
        raise NotImplementedError


class ImageServiceImpl(IImageService):
    image_repository: Annotated[IImageRepository, Provide[RepositoryContainer.image_repository]]
    label_repository: Annotated[ILabelRepository, Provide[RepositoryContainer.label_repository]]
    file_service: Annotated[IFileService, Provide[ServiceContainer.file_service]]

    async def get_image_by_id(self, image_id: UUID) -> Image:
        db_image = await self.image_repository.get_by_id(entity_id=image_id)
        if db_image is None:
            raise NotFoundError(f'No image with id {image_id} found.')
        return db_image

    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        file_bytes = await file.read()
        save_file = IFileService.SaveFile(name=file.filename, mime_type=file.content_type, data=file_bytes)
        metadata = await self.file_service.save_file(save_file)
        db_image = await self.image_repository.save(Image(
            created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
            name=file.filename,
            mime_type=file.content_type,
            save_path=metadata.path,
            user_id=user_id
        ))
        return db_image.id

    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        db_image = await self.get_image_by_id(image_id)
        await self.label_repository.assign_labels(db_image, label_ids)
