from abc import abstractmethod
from uuid import UUID

from src.data.model import Document, DocumentChunk
from src.repository import IRepository
from src.util import PagingParams, PagingWrapper


class IDocumentRepository(IRepository[UUID, Document]):

    @abstractmethod
    async def get_embedded(self, params: PagingParams) -> PagingWrapper[Document]:
        """
        Retrieve a paginated wrapper of `Document` objects based on the provided paging
        parameters by querying the database. Calculates both the total count of distinct
        documents and retrieves the specific page of documents as defined by the offset
        and limit in the paging parameters.

        :param params: An instance of PagingParams representing the pagination
            parameters such as offset and limit.
        :return: A paginated wrapper containing a list of Documents and associated
            metadata such as total count.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_unembedded(self, params: PagingParams) -> PagingWrapper[Document]:
        """
        Retrieve documents that do not have any associated chunks.

        This method queries the database to fetch a paginated list of documents
        that do not have any related `DocumentChunk`. It uses a count query to
        determine the total number of such documents and a separate query to
        retrieve the specified page of results based on the provided pagination
        parameters.

        :param params: Pagination parameters, specifying the `offset` and `limit`
            for the page of results to be retrieved.
        :return: A paginated wrapper containing the retrieved documents and
            pagination metadata.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_chunks(self, chunks: list[DocumentChunk]) -> None:
        """
        Deletes a list of DocumentChunk objects from the database.

        This function iterates over the provided list of chunks and deletes each
        chunk from the database using the active session created from the connection.
        The changes are committed after all deletions are performed.

        :param chunks: A list of DocumentChunk objects to be deleted.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
