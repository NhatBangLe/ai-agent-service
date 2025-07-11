import logging
from uuid import UUID

from .interface.thread import IThreadService
from ..data.dto import ThreadUpdate, ThreadCreate, OutputMessage
from ..data.model import Thread
from ..repository.interface.thread import IThreadRepository
from ..util import PagingWrapper, PagingParams


class ThreadServiceImpl(IThreadService):
    thread_repository: IThreadRepository
    _logger = logging.getLogger(__name__)

    def __init__(self, thread_repository: IThreadRepository):
        super().__init__()
        self.thread_repository = thread_repository

    async def get_all_threads_by_user_id(self, user_id: UUID, params: PagingParams) -> PagingWrapper[Thread]:
        return await self.thread_repository.get_all_by_user_id(user_id, params)

    async def get_thread_by_id(self, thread_id: UUID) -> Thread:
        db_thread = await self.thread_repository.get_by_id(thread_id)
        if db_thread is None:
            raise ValueError(f'Thread with id {thread_id} not found.')
        return db_thread

    async def get_all_messages_from_thread(self, thread_id: UUID, params: PagingParams) -> PagingWrapper[OutputMessage]:
        raise NotImplementedError

    async def create_thread(self, user_id: UUID, data: ThreadCreate) -> UUID:
        db_thread = await self.thread_repository.save(Thread(title=data.title, user_id=user_id))
        return db_thread.id

    async def update_thread(self, thread_id: UUID, data: ThreadUpdate) -> None:
        db_thread = await self.get_thread_by_id(thread_id)
        db_thread.title = data.title
        await self.thread_repository.save(db_thread)

    async def delete_thread_by_id(self, thread_id: UUID) -> Thread:
        db_thread = await self.get_thread_by_id(thread_id)
        await self.thread_repository.delete(db_thread)
        return db_thread
