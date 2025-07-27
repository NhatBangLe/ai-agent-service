from abc import ABC, abstractmethod
from os import PathLike
from uuid import UUID

from src.data.dto import DocumentCreate
from src.data.model import Document
from src.util import PagingParams, PagingWrapper


class IDocumentService(ABC):

    @abstractmethod
    async def get_document_by_id(self, document_id: UUID) -> Document:
        """
        Retrieve a document by its unique identifier.

        This asynchronous method fetches a document corresponding to the provided
        identifier from the document repository.

        :param document_id: The unique identifier of the document.
        :return: The document object retrieved from the repository.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises NotFoundError: If the document with the specified ID does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_document(self, data: DocumentCreate) -> Document:
        """
        Asynchronously saves an uploaded document file to the system, extracts its metadata,
        and stores its details in the database. The file's corresponding attributes, such
        as name, description, MIME type, and storage path, are recorded. If the file's name
        is absent or exceeds the allowed length, a default name is generated or truncated.

        :param data: The document data.
        :return: The saved document entry in the database.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises InvalidArgumentError: If the system does not support the file's MIME type.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document_by_id(self, document_id: UUID) -> Document:
        """
        Deletes a document by its unique identifier (UUID). The method interacts with the
        document repository to perform the deletion operation asynchronously.

        :param document_id: UUID of the document to be deleted.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises NotFoundError: If the document with the specified ID does not exist.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_document(self, document: Document) -> None:
        """
        Deletes a given document from the repository.

        This method performs an asynchronous operation to remove the specified
        document from the underlying storage or repository. It assumes the document
        already exists in the repository.

        :param document: The document to be removed.
        :raises NotFoundError: If the document with the specified ID does not exist.
        :raises NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_embedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        """
        Asynchronously retrieves embedded documents based on provided paging parameters.

        This function interacts with the document repository to fetch a paginated
        result of embedded documents. It uses the specified paging parameters to
        determine the scope and range of the pagination and returns a wrapper object
        that contains the resulting documents and additional pagination metadata.

        :param params: Paging parameters used for pagination of embedded documents.
        :return: A paging wrapper containing the embedded documents and pagination metadata.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_unembedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        """
        Fetches a paginated list of documents that are not embedded within another document.

        :param params: Holds paging details such as the page number, page size, and/or
            other criteria for pagination.
        :return: A wrapper containing the requested set of documents along with pagination
            metadata.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def embed_document(self, store_name: str, doc_id: UUID, chunk_ids: list[str]) -> None:
        """
        Asynchronously embeds a document into the specified vector store by adding
        corresponding chunks to the document. The chunks are linked using their
        respective chunk IDs, and the document is updated in the repository.

        :param store_name: Name of the vector store where the document should
            be embedded.
        :param doc_id: Unique identifier of the document to be embedded.
        :param chunk_ids: List of chunk IDs to be added to the document.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def unembed_document(self, doc_id: UUID) -> list[str]:
        """
        Asynchronously removes the embedding of a document from the vector store and
        deletes its chunks. The operation differs depending on whether the document
        source was uploaded by the user.

        This method retrieves the document by its ID, processes the associated chunks,
        and updates or deletes the document in the repository accordingly.

        Chunks that belong to the document are disassociated and deleted when necessary.
        If the document source is uploaded, it modifies the document's embedding flag
        and removes associated chunks. For other sources, the document is deleted entirely.

        :param doc_id: Unique identifier of the document to unembed
        :return: the List of chunk identifiers that were associated with the document.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def insert_external_document(self, store_name: str, file_path: str | PathLike[str]) -> None:
        """
        Inserts external documents into a specified store, processes their configuration,
        and updates the status in the configuration file. The function reads the provided
        JSON configuration file, validates it, and checks if the external data has already
        been configured. If not, the function processes each document specified in the
        configuration, converts the chunks to database objects, and commits the data to
        the database. Finally, the function updates the configuration file to mark the
        external data as configured.

        :param store_name: Name of the document store where the external documents should
                           be inserted.
        :param file_path: Path to the external documents configuration JSON file.
        :raises ValidationError: Raised if the JSON configuration file cannot be validated.
        :raises FileNotFoundError: Raised if the specified configuration file path does
                                   not exist.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
