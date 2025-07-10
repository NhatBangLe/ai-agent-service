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
from ..util import PagingWrapper, PagingParams
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
        :return: The image corresponding to the provided identifier.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        :raises NotFoundError: If the image with the specified ID does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_images_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        """
        Fetch images associated with specific label IDs using pagination.

        This asynchronous method retrieves a list of images that are associated
        with the provided label IDs. It uses pagination to limit and offset
        the results as specified in the `params` argument.

        :param params: Pagination parameters including page size and page number.
        :param label_ids: A list of label IDs for which images are to be fetched.
        :return: A PagingWrapper object containing the paginated list of images.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_unlabeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        """
        Retrieves all unlabeled images using the specified paging parameters. This asynchronous method calls
        the corresponding repository to fetch a paginated list of unlabeled images.

        :param params: The paging parameters used for pagination, provided as a `PagingParams` object.
        :return: Returns a `PagingWrapper` containing the paginated result of unlabeled images.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_labeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        """
        Fetches a paginated list of labeled images from the image repository.

        This asynchronous method retrieves a collection of images that are already
        labeled, based on the given pagination parameters.

        :param params: Pagination parameters including page size and page number.
        :return: A paginated wrapper containing labeled images.
        :raises NotImplementedError: If the method is not implemented in the subclass.
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
        :param file: The file object representing the image to be saved.
        :return: A unique identifier for the saved image.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_image_by_id(self, image_id: UUID) -> Image:
        """
        Deletes an image by its unique identifier.

        This async method interacts with the image repository to delete an image based
        on its ID. If no image matches the provided ID, a `NotFoundError` is raised.
        After successfully deleting the image from the repository, it proceeds to
        delete the corresponding file using the file service. The deleted image
        information is then returned.

        :param image_id: The unique identifier of the image to be deleted.
        :return: The details of the deleted image including its metadata.
        :raises NotFoundError: If no image is found with the specified ID.
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

    async def get_images_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        return await self.image_repository.get_all_by_label_ids(params, label_ids)

    async def get_unlabeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        return await self.image_repository.get_all_unlabeled(params)

    async def get_labeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        return await self.image_repository.get_all_labeled(params)

    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        file_bytes = await file.read()
        save_file = IFileService.SaveFile(name=file.filename, mime_type=file.content_type, data=file_bytes)
        saved_file_id = await self.file_service.save_file(save_file)
        db_image = await self.image_repository.save(Image(user_id=user_id, file_id=saved_file_id))
        return db_image.id

    async def delete_image_by_id(self, image_id: UUID) -> Image:
        deleted_image = await self.image_repository.delete_by_id(image_id)
        if deleted_image is None:
            raise NotFoundError(f'Image with id {image_id} not found.')
        await self.file_service.delete_file_by_id(deleted_image.file_id)
        return deleted_image

    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        db_image = await self.get_image_by_id(image_id)
        await self.label_repository.assign_labels(db_image, label_ids)
