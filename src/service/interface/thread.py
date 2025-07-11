from abc import ABC, abstractmethod
from uuid import UUID

from src.data.dto import OutputMessage, ThreadCreate, ThreadUpdate
from src.data.model import Thread
from src.util import PagingParams, PagingWrapper


class IThreadService(ABC):

    @abstractmethod
    async def get_all_threads_by_user_id(self, user_id: UUID, params: PagingParams) -> PagingWrapper[Thread]:
        """
        Fetches all threads associated with a given user ID. The function uses
        the provided parameters for pagination to efficiently retrieve
        a subset of threads rather than fetching all threads at once.

        :param user_id: Unique identifier of the user whose threads are to be fetched.
        :param params: Pagination parameters, including page size and offset.
        :return: A paginated wrapper object containing the list of threads
                 associated with the specified user.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_thread_by_id(self, thread_id: UUID) -> Thread:
        """
        Retrieve a thread by its unique identifier.

        This asynchronous method fetches a thread from the repository using the
        provided thread ID. If the thread does not exist, an exception is raised.

        :param thread_id: The unique identifier of the thread to retrieve.
        :returns: The thread corresponding to the given identifier.
        :raises ValueError: If the thread with the provided ID is not found.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_all_messages_from_thread(self, thread_id: UUID, params: PagingParams) -> PagingWrapper[OutputMessage]:
        raise NotImplementedError

    @abstractmethod
    async def create_thread(self, user_id: UUID, data: ThreadCreate) -> UUID:
        """
        Creates a new thread in the system and stores it in the database. This method uses a thread
        repository to save the thread with the user's ID and the provided thread data. The method
        is asynchronous and returns the unique identifier of the newly created thread.

        :param user_id: The unique identifier of the user creating the thread.
        :param data: The data is required to create a new thread, including properties such as the title.
        :return: The unique identifier of the newly created thread.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def update_thread(self, thread_id: UUID, data: ThreadUpdate) -> None:
        """
        Updates an existing thread with provided data. Fetches the thread by its ID,
        modifies it, and saves the updated thread to the repository.

        :param thread_id: UUID of the thread that is being updated.
        :param data: Data containing the updated title for the thread.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_thread_by_id(self, thread_id: UUID) -> Thread:
        """
        Deletes a thread from the database by its unique identifier.

        This method retrieves a thread from the database using its unique
        identifier, deletes it through the thread repository, and then
        returns the deleted thread instance.

        :param thread_id: The unique identifier of the thread to be deleted.
        :return: The thread instance that was deleted.
        :raises NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError
