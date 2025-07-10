from abc import abstractmethod
from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import IRepository, RepositoryImpl
from ..data.model import Document, DocumentChunk
from ..util import PagingParams, PagingWrapper


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


# noinspection PyTypeChecker
class DocumentRepositoryImpl(IDocumentRepository, RepositoryImpl):

    async def get_embedded(self, params: PagingParams) -> PagingWrapper[Document]:
        with self.connection.create_session() as session:
            count_stmt = (select(func.count(func.distinct(Document.id)))
                          .select_from(Document)
                          .join(DocumentChunk, DocumentChunk.document_id == Document.id))
            exec_stmt = (select(Document)
                         .distinct()
                         .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                         .offset((params.offset * params.limit))
                         .limit(params.limit)
                         .order_by(Document.created_at))
            return PagingWrapper.get_paging(
                params=params,
                execute_statement=exec_stmt,
                count_statement=count_stmt,
                session=session)

    async def get_unembedded(self, params: PagingParams) -> PagingWrapper[Document]:
        with self.connection.create_session() as session:
            count_stmt = (select(func.count())
                          .outerjoin_from(Document, DocumentChunk, DocumentChunk.document_id == Document.id)
                          .where(DocumentChunk.document_id == None))
            exec_stmt = (select(Document)
                         .join(DocumentChunk, DocumentChunk.document_id == Document.id, isouter=True)
                         .where(DocumentChunk.document_id == None)
                         .offset((params.offset * params.limit))
                         .limit(params.limit)
                         .order_by(Document.created_at))
            return PagingWrapper.get_paging(
                params=params,
                execute_statement=exec_stmt,
                count_statement=count_stmt,
                session=session)

    async def delete_chunks(self, chunks: list[DocumentChunk]) -> None:
        with self.connection.create_session() as session:
            for chunk in chunks:
                session.delete(chunk)
            session.commit()
