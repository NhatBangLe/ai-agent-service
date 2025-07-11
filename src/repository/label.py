import datetime
from uuid import UUID

from sqlmodel import select

from . import RepositoryImpl
from .interface.label import ILabelRepository
from ..data.model import Label, LabeledImage, Image
from ..util.constant import DEFAULT_TIMEZONE


# noinspection PyTypeChecker
class LabelRepositoryImpl(ILabelRepository, RepositoryImpl):

    async def get_all_by_image_id(self, image_id: UUID) -> list[Label]:
        with self.connection.create_session() as session:
            statement = (select(Label)
                         .join(LabeledImage, LabeledImage.label_id == Label.id)
                         .where(LabeledImage.image_id == image_id)
                         .order_by(LabeledImage.created_at))
            results = session.exec(statement)
            return list(results.all())

    async def get_all(self) -> list[Label]:
        with self.connection.create_session() as session:
            statement = select(Label)
            results = session.exec(statement)
            return list(results.all())

    async def get_by_name(self, name: str) -> Label | None:
        with self.connection.create_session() as session:
            stmt = select(Label).where(Label.name == name).limit(1)
            label = session.exec(stmt).one_or_none()
            return label

    # noinspection PyUnresolvedReferences
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
