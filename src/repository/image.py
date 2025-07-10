from abc import abstractmethod
from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Image, LabeledImage
from ..util import PagingParams, PagingWrapper


class IImageRepository(IRepository[UUID, Image]):

    @abstractmethod
    async def get_all_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        """
        Fetches a paginated list of images based on the provided label IDs. The query ensures
        that only images containing all the specified label IDs are retrieved. The function
        uses a subquery to filter images by label IDs and then constructs statements
        for counting and fetching the paginated results.

        :param params: Pagination details, including offset and limit.
        :param label_ids: A list of label IDs to filter the images.
        :return: A paginated wrapper containing the list of images and the total count.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all_unlabeled(self, params: PagingParams) -> PagingWrapper[Image]:
        """
        Retrieves all images that are not yet labeled based on the specified paging parameters.
        Returns a `PagingWrapper` containing the list of unlabeled images and metadata about
        the paging.

        :param params: The paging parameters including offset and limit are used to filter the
                       results. The results will be offset by the given value and limited
                       to the specified count.
        :return: A `PagingWrapper` object containing the list of unlabeled images and additional
                 paging metadata.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all_labeled(self, params: PagingParams) -> PagingWrapper[Image]:
        """
        Retrieves all labeled images from the database with paging parameters. This
        method queries the database to count and fetch a limited set of distinct
        `Image` objects that are associated with labels. The query supports offset
        and limit for pagination and orders the results based on the `created_at`
        attribute of the `Image` objects.

        :param params: Paging parameters, including limit and offset, are used to
            paginate the results.
        :return: A `PagingWrapper` instance containing the paginated list of `Image`
            objects and additional paging metadata.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


# noinspection PyComparisonWithNone,PyTypeChecker
class ImageRepositoryImpl(IImageRepository, RepositoryImpl):

    # noinspection PyUnresolvedReferences
    async def get_all_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        with self.connection.create_session() as session:
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
        with self.connection.create_session() as session:
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
