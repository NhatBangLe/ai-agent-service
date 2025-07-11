from abc import abstractmethod
from uuid import UUID

from src.data.model import Label, Image
from src.repository import IRepository


class ILabelRepository(IRepository[int, Label]):

    @abstractmethod
    async def get_all_by_image_id(self, image_id: UUID) -> list[Label]:
        """
        Get all labels associated with a specific image identifier.

        It retrieves a list of labels related to a given image based on the provided image ID.

        :param image_id: The unique identifier for the image.
        :return: A list of Label objects associated with the provided image ID.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all(self) -> list[Label]:
        """
        It retrieves all Label instances asynchronously.
        The specific implementation is left to the subclass.

        :return: A list of Label instances.
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_by_name(self, name: str) -> Label | None:
        """
        The method is asynchronous in nature and expects the implementation to provide a
        mechanism to either return the corresponding label object or `None` if no
        such label exists.

        :param name: The name of the label to be retrieved.
        :return: The label matching the provided name or `None` if not found.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def assign_labels(self, image: Image, label_ids: list[int]) -> None:
        """
        Assign labels to a given image asynchronously.

        It associates a set of label IDs with a given image.

        :param image: The image object to which labels will be assigned.
        :param label_ids: A list of label IDs that need to be assigned to the image.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError
