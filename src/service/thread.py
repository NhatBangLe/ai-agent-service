import logging

from sqlmodel import select

from .interface.thread import IThreadService
from ..data.model import Thread, User, File
from ..repository.interface.thread import IThreadRepository
from ..util.error import NotFoundError


class ThreadServiceImpl(IThreadService):
    _thread_repository: IThreadRepository
    _logger = logging.getLogger(__name__)

    def __init__(self, thread_repository: IThreadRepository):
        super().__init__()
        self._thread_repository = thread_repository

    async def get_all_threads_by_user_id(self, user_id, params):
        return await self._thread_repository.get_all_by_user_id(user_id, params)

    async def get_thread_by_id(self, thread_id):
        db_thread = await self._thread_repository.get_by_id(thread_id)
        if db_thread is None:
            raise NotFoundError(f'Thread with id {thread_id} not found.')
        return db_thread

    async def create_thread(self, user_id, data):
        with await self._thread_repository.get_session() as session:
            user = session.get(User, user_id)
            if user is None:
                user = User(id=user_id)
            db_thread = Thread(title=data.title, user=user)
            session.add(db_thread)
            session.commit()
            session.refresh(db_thread, ["id"])
            return db_thread

    async def update_thread(self, thread_id, data):
        db_thread = await self.get_thread_by_id(thread_id)
        db_thread.title = data.title
        await self._thread_repository.save(db_thread)

    async def delete_thread_by_id(self, thread_id):
        db_thread = await self.get_thread_by_id(thread_id)
        await self._thread_repository.delete(db_thread)
        return db_thread

    # noinspection PyUnresolvedReferences,PyTypeChecker
    async def add_attachments(self, thread_id, file_ids):
        with await self._thread_repository.get_session() as session:
            db_thread = session.get(Thread, thread_id)
            if db_thread is None:
                raise NotFoundError(f'Cannot add attachments because thread with id {thread_id} not found.')

            stmt = select(File).where(File.id.in_(file_ids))
            files: list[File] = list(session.exec(stmt).all())
            db_thread.attachments += files
            session.add(db_thread)
            session.commit()

    async def delete_attachment_by_id(self, attachment_id) -> None:
        with await self._thread_repository.get_session() as session:
            attachment: File | None = session.get(File, attachment_id)
            if attachment is None:
                raise NotFoundError(f'Cannot delete attachment because it is not found.')
            attachment.thread = None
            session.add(attachment)
            session.commit()
