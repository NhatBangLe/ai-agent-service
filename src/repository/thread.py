from abc import abstractmethod
from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Thread
from ..util import PagingParams, PagingWrapper


class IThreadRepository(IRepository[UUID, Thread]):

    @abstractmethod
    async def get_all_by_user_id(self, user_id: UUID, params: PagingParams) -> PagingWrapper[Thread]:
        """
        Retrieve all threads created by a specific user.

        This asynchronous method retrieves paginated threads associated with a particular
        user ID from the database. It uses the provided paging parameters to limit and
        offset the query, ensuring proper pagination of the results.

        :param user_id: The unique identifier of the target user whose threads are being
            retrieved.
        :param params: An instance of PagingParams containing pagination details such as
            offset and limit.
        :return: A PagingWrapper instance containing the paginated thread data.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class ThreadRepositoryImpl(IThreadRepository, RepositoryImpl):

    async def get_all_by_user_id(self, user_id: UUID, params: PagingParams):
        with self.connection.create_session() as session:
            count_statement = (select(func.count())
                               .where(Thread.user_id == user_id))
            execute_statement = (select(Thread)
                                 .where(Thread.user_id == user_id)
                                 .offset(params.offset * params.limit)
                                 .limit(params.limit)
                                 .order_by(Thread.created_at))
            return PagingWrapper.get_paging(params, count_statement, execute_statement, session)
