from abc import ABC, abstractmethod
from os import PathLike
from uuid import UUID

from src.data.dto import LabelCreate, LabelUpdate
from src.data.model import Label


class ILabelService(ABC):

    @abstractmethod
    async def get_all_labels(self) -> list[Label]:
        """
        Retrieves all labels available in the repository.

        This method fetches all labels from the label repository and returns them
        as a list of `Label` objects. It acts as an interface to interact with the
        underlying data storage to retrieve all stored label entities.

        :return: A list of all `Label` objects fetched from the repository.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_label_by_id(self, label_id: int) -> Label:
        """
        Gets the label corresponding to the provided label ID.

        This method must be implemented in a subclass to return the label
        associated with the specified label ID. The returned label typically
        contains information relevant to the ID, such as name and metadata.
        The implementation of this method is expected to be asynchronous.

        :param label_id: Unique identifier of the label to retrieve.
        :type label_id: int
        :return: A label object corresponding to the provided ID.
        :rtype: Label
        :raises NotImplementedError: If the method is not implemented by a subclass.
        :raises NotFoundError: If the label with the specified ID does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_labels_by_image_id(self, image_id: UUID) -> list[Label]:
        """
        Retrieve all labels associated with a specific image ID.

        This asynchronous method interacts with the label repository to fetch
        all the labels linked to the provided image ID. It ensures that only
        labels that match the given identifier are returned as part of the
        result.

        :param image_id: Unique identifier for the image whose labels need to
            be retrieved.
        :return: A list of Label objects associated with the specified image ID.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def create_label(self, label: LabelCreate) -> int:
        """
        Creates a new label entry in the system asynchronously.

        This abstract method is intended to be implemented by a subclass
        responsible for handling the creation of a label. The method takes
        details about the label to be created and returns an identifier
        of the newly created label. It raises a NotImplementedError if
        called directly.

        :param label: An instance of `LabelCreate` containing details of the label to
            be created.
        :return: The unique identifier of the newly created label.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        :raises InvalidArgumentError: If the label name already exists.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_label(self, label_id: int, label: LabelUpdate) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_label_by_name(self, label_name: str) -> Label:
        """
        Deletes a label by its name asynchronously if it exists in the repository.

        This method attempts to find a label with the specified name in the label
        repository and deletes it. If no label with the given name is found, an
        exception is raised to indicate that the operation could not be completed.

        :param label_name: The name of the label to be deleted.
        :return: The label that was successfully deleted.
        :raises NotFoundError: If no label with the specified name is found.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_label_by_id(self, label_id: int) -> Label:
        """
        Deletes a label with the specified ID. If the label with the given ID is not
        found, raises a NotFoundError.

        :param label_id: ID of the label to be deleted
        :return: The deleted Label object
        :raises NotFoundError: If no label is found with the provided ID.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def insert_predefined_output_classes(self, config_file_path: str | PathLike[str]):
        """
        Initializes the application's database with predefined labels from a configuration file.
        This process is designed to run only once to prevent duplicate data insertion.

        Args:
            config_file_path: A absolute path to the configuration file containing the predefined labels.

        Raises:
            FileNotFoundError: If the configuration file specified by `config.output_config_path`
                               does not exist.
            IOError: If there's an issue reading from or writing to the configuration file.
            DatabaseError: If there's an issue interacting with the database (e.g., adding labels, committing).
            ValidationError: If the content of the configuration file does not conform to the
                             `RecognizerOutput` model's expected structure.
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
