from abc import abstractmethod
from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Image, LabeledImage
from ..util import PagingParams, PagingWrapper


class IImageRepository(IRepository[UUID, Image]):

    @abstractmethod
    async def get_all_by_label_id(self, label_id: int, params: PagingParams) -> PagingWrapper[Image]:
        raise NotImplementedError

    @abstractmethod
    async def get_all_unlabeled(self, params: PagingParams) -> PagingWrapper[Image]:
        raise NotImplementedError

    @abstractmethod
    async def get_all_labeled(self, params: PagingParams) -> PagingWrapper[Image]:
        raise NotImplementedError


# noinspection PyComparisonWithNone,PyTypeChecker
class ImageRepositoryImpl(IImageRepository, RepositoryImpl):

    async def get_all_by_label_id(self, label_id: int, params: PagingParams) -> PagingWrapper[Image]:
        with self.connection.create_session() as session:
            count_statement = (select(func.count())
                               .outerjoin_from(Image, LabeledImage, LabeledImage.image_id == Image.id)
                               .where(LabeledImage.label_id == label_id))

            statement = (select(Image)
                         .distinct()
                         .join(LabeledImage, LabeledImage.image_id == Image.id)
                         .where(LabeledImage.label_id == label_id)
                         .offset((params.offset * params.limit))
                         .limit(params.limit)
                         .order_by(Image.created_at))
            return PagingWrapper.get_paging(
                params=params,
                count_statement=count_statement,
                execute_statement=statement,
                session=session
            )

    async def get_all_unlabeled(self, params: PagingParams) -> PagingWrapper[Image]:
        with self.connection.create_session() as session:
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

    async def get_all_labeled(self, params: PagingParams) -> PagingWrapper[Image]:
        pass
