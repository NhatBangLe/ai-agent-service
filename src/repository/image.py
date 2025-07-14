from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import RepositoryImpl
from .interface.image import IImageRepository
from ..data.model import Image, LabeledImage, Label
from ..util import PagingParams, PagingWrapper


# noinspection PyComparisonWithNone,PyTypeChecker
class ImageRepositoryImpl(IImageRepository, RepositoryImpl):

    async def get_by_id(self, entity_id: UUID) -> Image | None:
        with self._connection.create_session() as session:
            entity = session.get(Image, entity_id)
            return entity

    # noinspection PyUnresolvedReferences
    async def get_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        with self._connection.create_session() as session:
            subquery = (select(LabeledImage.image_id)
                        .where(LabeledImage.label_id.in_(label_ids))
                        .group_by(LabeledImage.image_id)
                        .having(func.count(LabeledImage.label_id) == len(params.label_ids))
                        .subquery())

            count_statement = (select(func.count())
                               .select_from(Image)
                               .where(Image.id.in_(select(subquery.c.image_id))))

            statement = (select(Image)
                         .where(Image.id.in_(select(subquery.c.image_id)))
                         .offset(params.offset * params.limit)
                         .limit(params.limit)
                         .order_by(Image.created_at))

            return PagingWrapper.get_paging(
                params=params,
                count_statement=count_statement,
                execute_statement=statement,
                session=session)

    async def get_unlabeled(self, params: PagingParams) -> PagingWrapper[Image]:
        with self._connection.create_session() as session:
            count_statement = (select(func.count())
                               .outerjoin_from(Image, LabeledImage, LabeledImage.image_id == Image.id)
                               .where(LabeledImage.label_id == None))

            statement = (select(Image)
                         .join(LabeledImage, LabeledImage.image_id == Image.id, isouter=True)
                         .where(LabeledImage.label_id == None)
                         .offset(params.offset)
                         .limit(params.limit)
                         .order_by(Image.created_at))
            return PagingWrapper.get_paging(
                params=params,
                count_statement=count_statement,
                execute_statement=statement,
                session=session
            )

    async def get_labeled(self, params: PagingParams) -> PagingWrapper[Image]:
        with self._connection.create_session() as session:
            count_statement = (select(func.count(func.distinct(Image.id)))
                               .select_from(Image)
                               .join(LabeledImage, LabeledImage.image_id == Image.id))
            statement = (select(Image)
                         .distinct()
                         .join(LabeledImage, LabeledImage.image_id == Image.id)
                         .offset((params.offset * params.limit))
                         .limit(params.limit)
                         .order_by(Image.created_at))
            return PagingWrapper.get_paging(
                params=params,
                count_statement=count_statement,
                execute_statement=statement,
                session=session)

    async def get_all_by_label_id(self, label_id: int) -> list[Image]:
        with self._connection.create_session() as session:
            get_all_images_stmt = (select(Image)
                                   .join(LabeledImage, LabeledImage.image_id == Image.id)
                                   .where(LabeledImage.label_id == label_id))
            return list(session.exec(get_all_images_stmt).all())

    async def get_all_images_with_labels(self) -> list[tuple[Label, Image]]:
        with self._connection.create_session() as session:
            get_all_used_labels_stmt = (select(Label, Image)
                                        .join(LabeledImage, LabeledImage.label_id == Label.id)
                                        .join(Image, LabeledImage.image_id == Image.id))
            return list(session.exec(get_all_used_labels_stmt).all())
