from abc import ABC, abstractmethod
from uuid import UUID

from src.data.dto import ImageCreate
from src.data.model import Image
from src.util import PagingParams, PagingWrapper


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
    async def save_image(self, data: ImageCreate) -> UUID:
        """
        Saves an image uploaded by the user asynchronously. The method associates the
        uploaded file with the provided user ID and returns the unique identifier for the
        stored image. It is meant to be implemented by subclasses and raises a
        NotImplementedError if called directly from the base class.

        :param data: The file object representing the image to be saved.
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
    async def delete_image(self, image: Image) -> None:
        """
        Deletes the specified image using the image repository. This asynchronous method
        ensures the image is removed effectively from the storage or database managed
        by the image repository.

        :param image: The image object to be deleted.
        :return: This method does not return any value.
        :raises NotFoundError: If no image is found with the specified ID.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def assign_labels_by_label_ids(self, image_id: UUID, label_ids: list[int]) -> None:
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

    @abstractmethod
    async def assign_labels_by_label_names(self, image_id: UUID, label_names: list[str]) -> None:
        """
        Assigns a list of labels to a specific image.

        Assigns the given labels to the specified image
        using its unique identifier. This is an abstract method and must be
        implemented in a subclass.

        :param image_id: The unique identifier for the image to assign labels to.
        :param label_names: A list of unique label names for the labels to be assigned
            to the image.
        :raises NotImplementedError: If the method is not implemented.
        :raises NotFoundError: If the image with the specified ID does not exist.
        """
        raise NotImplementedError
