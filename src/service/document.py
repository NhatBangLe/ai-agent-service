import asyncio
import datetime
import logging
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide
from fastapi import UploadFile

from .container import ServiceContainer
from .file import IFileService
from ..config.model.data import ExternalDocumentConfiguration
from ..data.base_model import DocumentSource
from ..data.model import DocumentChunk, Document
from ..repository.container import RepositoryContainer
from ..repository.document import IDocumentRepository
from ..util import PagingWrapper, PagingParams
from ..util.constant import DEFAULT_TIMEZONE, SUPPORTED_DOCUMENT_TYPE_DICT
from ..util.error import InvalidArgumentError, NotFoundError


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
        """
        raise NotImplementedError

    @abstractmethod
    async def save_document(self, file: UploadFile, description: str | None) -> UUID:
        """
        Asynchronously saves an uploaded document file to the system, extracts its metadata,
        and stores its details in the database. The file's corresponding attributes, such
        as name, description, MIME type, and storage path, are recorded. If the file's name
        is absent or exceeds the allowed length, a default name is generated or truncated.

        :param file: The uploaded document file to be saved.
        :param description: An optional description for the document being uploaded.
        :return: The unique identifier (UUID) of the saved document entry in the database.
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


class DocumentServiceImpl(IDocumentService):
    document_repository: Annotated[IDocumentRepository, Provide[RepositoryContainer.document_repository]]
    file_service: Annotated[IFileService, Provide[ServiceContainer.file_service]]
    _logger = logging.getLogger(__name__)

    async def get_document_by_id(self, document_id: UUID) -> Document:
        return await self.document_repository.get_by_id(entity_id=document_id)

    async def save_document(self, file: UploadFile, description: str | None) -> UUID:
        file_bytes = await file.read()

        mime_type = file.content_type
        ext = SUPPORTED_DOCUMENT_TYPE_DICT[mime_type]
        if ext is None:
            raise InvalidArgumentError(f'Unsupported MIME type: {mime_type}')

        save_file = IFileService.SaveFile(name=file.filename, mime_type=mime_type, data=file_bytes)
        metadata = await self.file_service.save_file(save_file)

        max_name_len = 255
        file_name = file.filename
        if file_name is None:
            current_datetime = datetime.datetime.now(DEFAULT_TIMEZONE)
            file_name = f'file-{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{ext}'
        if len(file_name) > max_name_len:
            file_name = file_name.split('.')[0]
            max_len_value = max_name_len - len(ext)
            if len(file_name) > max_len_value:
                file_name = file_name[:max_len_value]
        db_doc = await self.document_repository.save(Document(created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
                                                              description=description,
                                                              name=file_name,
                                                              mime_type=mime_type,
                                                              save_path=metadata.path,
                                                              source=DocumentSource.UPLOADED))

        return db_doc.id

    async def delete_document_by_id(self, document_id: UUID) -> Document:
        deleted_document = await self.document_repository.delete_by_id(document_id)
        if deleted_document is None:
            raise NotFoundError(f'Document with id {document_id} not found.')
        return deleted_document

    async def get_embedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        return await self.document_repository.get_embedded(params)

    async def get_unembedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        return await self.document_repository.get_unembedded(params)

    async def embed_document(self, store_name: str, doc_id: UUID, chunk_ids: list[str]) -> None:
        db_doc = await self.get_document_by_id(doc_id)
        db_doc.embed_to_vs = store_name
        db_doc.chunks += [DocumentChunk(id=chunk_id) for chunk_id in chunk_ids]
        await self.document_repository.save(db_doc)

    async def unembed_document(self, doc_id: UUID) -> list[str]:
        db_doc = await self.get_document_by_id(doc_id)

        db_chunks = db_doc.chunks
        chunk_ids = [chunk.id for chunk in db_chunks]

        if db_doc.source == DocumentSource.UPLOADED:
            db_doc.embed_to_vs = None
            db_doc.chunks = []
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.document_repository.delete_chunks(db_chunks))
                tg.create_task(self.document_repository.save(db_doc))
        else:
            await self.document_repository.delete(db_doc)

        return chunk_ids

    async def insert_external_document(self, store_name: str, file_path: str | PathLike[str]) -> None:
        file_path = Path(file_path)
        json_bytes = file_path.read_bytes()
        config = ExternalDocumentConfiguration.model_validate_json(json_bytes)
        if config.is_configured:
            self._logger.debug("External data are already configured. Skipping...")
            return

        docs: list[Document] = []
        for d in config.documents:
            db_chunks = [DocumentChunk(id=str(chunk_id)) for chunk_id in d.chunk_ids]
            db_doc = Document(
                created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
                name=d.name,
                source=DocumentSource.EXTERNAL,
                embed_to_vs=store_name,
                chunks=db_chunks)
            docs.append(db_doc)
        await self.document_repository.save_all(docs)

        self._logger.debug(f'Updating external data configuration file with configured status: is_configured = True...')
        with open(file_path, 'w'):  # Clear old content
            pass
        config.is_configured = True  # Mark as configured
        file_path.write_text(config.model_dump_json(indent=2))
