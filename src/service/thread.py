import logging
from uuid import UUID

from .interface.thread import IThreadService
from ..data.dto import ThreadUpdate, ThreadCreate
from ..data.model import Thread, User
from ..repository.interface.thread import IThreadRepository
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError


class ThreadServiceImpl(IThreadService):
    _thread_repository: IThreadRepository
    _logger = logging.getLogger(__name__)

    def __init__(self, thread_repository: IThreadRepository):
        super().__init__()
        self._thread_repository = thread_repository

    async def get_all_threads_by_user_id(self, user_id: UUID, params: PagingParams) -> PagingWrapper[Thread]:
        return await self._thread_repository.get_all_by_user_id(user_id, params)

    async def get_thread_by_id(self, thread_id: UUID) -> Thread:
        db_thread = await self._thread_repository.get_by_id(thread_id)
        if db_thread is None:
            raise NotFoundError(f'Thread with id {thread_id} not found.')
        return db_thread

    async def get_all_messages_from_thread(self, thread_id: UUID, params: PagingParams) -> PagingWrapper:
        raise NotImplementedError

    async def create_thread(self, user_id: UUID, data: ThreadCreate) -> UUID:
        with await self._thread_repository.get_session() as session:
            user = session.get(User, user_id)
            if user is None:
                user = User(id=user_id)
            db_thread = Thread(title=data.title, user=user)
            session.add(db_thread)
            session.commit()
            session.refresh(db_thread, ["id"])
            return db_thread.id

    async def update_thread(self, thread_id: UUID, data: ThreadUpdate) -> None:
        db_thread = await self.get_thread_by_id(thread_id)
        db_thread.title = data.title
        await self._thread_repository.save(db_thread)

    async def delete_thread_by_id(self, thread_id: UUID) -> Thread:
        db_thread = await self.get_thread_by_id(thread_id)
        await self._thread_repository.delete(db_thread)
        return db_thread
