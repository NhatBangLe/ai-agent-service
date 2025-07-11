from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import RepositoryImpl
from .interface.thread import IThreadRepository
from ..data.model import Thread
from ..util import PagingParams, PagingWrapper


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
