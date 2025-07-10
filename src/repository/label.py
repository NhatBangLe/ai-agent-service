from abc import abstractmethod
from uuid import UUID

from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Label, LabeledImage, Image
from ..util import PagingParams


class ILabelRepository(IRepository[int, Label]):

    @abstractmethod
    async def get_all_by_image_id(self, image_id: UUID, params: PagingParams) -> list[Label]:
        """
        Get all labels associated with a specific image identifier.

        This is an abstract method that should be implemented by a subclass. It retrieves
        a list of labels related to a given image based on the provided image ID. Results
        can be influenced by optional pagination parameters.

        :param image_id: The unique identifier for the image.
        :param params: An instance of PagingParams to control pagination of the results.
        :return: A list of Label objects associated with the provided image ID.
        :rtype: list[Label]
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all(self) -> list[Label]:
        """
        This is an abstract method that subclasses must implement. It retrieves
        all Label instances asynchronously. The specific implementation is left to
        the subclass.

        :return: A list of Label instances.
        :rtype: list[Label]
        :raises NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_by_name(self, name: str) -> Label | None:
        """
        This is an abstract method that retrieves a label by its name. The method
        is asynchronous in nature and expects the implementation to provide a
        mechanism to either return the corresponding label object or `None` if no
        such label exists.

        :param name: The name of the label to be retrieved.
        :type name: str
        :return: The label matching the provided name or `None` if not found.
        :rtype: Label | None
        :raises NotImplementedError: Indicates that the method is required to
            be implemented in a subclass before instantiation.
        """
        raise NotImplementedError

    @abstractmethod
    async def assign_labels(self, image: Image, label_ids: list[int]) -> None:
        """
        Assign labels to a given image asynchronously.

        This method is an abstract method and must be implemented in a subclass.
        It associates a set of label IDs with a given image.

        :param image: The image object to which labels will be assigned.
        :type image: Image
        :param label_ids: A list of label IDs that need to be assigned to the image.
        :type label_ids: list[int]
        :return: None
        """
        raise NotImplementedError


# noinspection PyComparisonWithNone,PyTypeChecker,PyUnresolvedReferences
class LabelRepositoryImpl(ILabelRepository, RepositoryImpl):

    async def get_all_by_image_id(self, image_id: UUID, params: PagingParams) -> list[Label]:
        with self.connection.create_session() as session:
            statement = (select(Label)
                         .join(LabeledImage, LabeledImage.label_id == Label.id)
                         .where(LabeledImage.image_id == image_id)
                         .order_by(LabeledImage.created_at)
                         .offset(params.offset)
                         .limit(params.limit))
            results = session.exec(statement)
            return list(results.all())

    async def get_all(self) -> list[Label]:
        with self.connection.create_session() as session:
            statement = select(Label)
            results = session.exec(statement)
            return list(results.all())

    async def get_by_name(self, name: str) -> Label | None:
        with self.connection.create_session() as session:
            label = (session.exec(select(Label).where(Label.name == name).limit(1))
                     .one_or_none())
            return label

    async def assign_labels(self, image: Image, label_ids: list[int]):
        with self.connection.create_session() as session:
            statement = select(Label).where(Label.id.in_(label_ids))
            matched_labels = session.exec(statement).all()
            for label in matched_labels:
                db_labeled_image = LabeledImage(
                    label=label, image=image,
                    created_at=datetime.datetime.now(DEFAULT_TIMEZONE))
                session.add(db_labeled_image)
            session.commit()
