from abc import abstractmethod
from uuid import UUID

from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Label, LabeledImage, Image
from ..util import PagingParams


class ILabelRepository(IRepository[int, Label]):

    @abstractmethod
    async def get_all_by_image_id(self, image_id: UUID, params: PagingParams) -> list[Label]:
        raise NotImplementedError

    @abstractmethod
    async def get_all(self) -> list[Label]:
        raise NotImplementedError

    @abstractmethod
    async def assign_labels(self, image: Image, label_ids: list[int]) -> None:
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
