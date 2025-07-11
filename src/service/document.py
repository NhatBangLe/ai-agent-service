import asyncio
import datetime
import logging
from os import PathLike
from pathlib import Path
from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide
from fastapi import UploadFile

from .container import ServiceContainer
from .interface.file import IFileService
from .interface.document import IDocumentService
from ..config.model.data import ExternalDocumentConfiguration
from ..data.base_model import DocumentSource
from ..data.model import DocumentChunk, Document
from ..repository.container import RepositoryContainer
from ..repository.document import IDocumentRepository
from ..util import PagingWrapper, PagingParams
from ..util.constant import DEFAULT_TIMEZONE, SUPPORTED_DOCUMENT_TYPE_DICT
from ..util.error import InvalidArgumentError, NotFoundError


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
        file_id = await self.file_service.save_file(save_file)

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

        db_doc = await self.document_repository.save(Document(description=description,
                                                              name=file_name,
                                                              file_id=file_id,
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
