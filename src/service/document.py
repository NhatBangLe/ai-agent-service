import logging
from os import PathLike
from pathlib import Path
from uuid import UUID

from .interface.document import IDocumentService
from ..config.model.data import ExternalDocumentConfiguration
from ..data.base_model import DocumentSource
from ..data.dto import DocumentCreate
from ..data.model import DocumentChunk, Document
from ..repository.interface.document import IDocumentRepository
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError


class DocumentServiceImpl(IDocumentService):
    _document_repository: IDocumentRepository
    _logger = logging.getLogger(__name__)

    def __init__(self, document_repository: IDocumentRepository):
        super().__init__()
        self._document_repository = document_repository

    async def get_document_by_id(self, document_id: UUID) -> Document:
        doc = await self._document_repository.get_by_id(entity_id=document_id)
        if doc is None:
            raise NotFoundError(f'Document with id {document_id} not found.')
        return doc

    async def save_document(self, data: DocumentCreate) -> UUID:
        db_doc = await self._document_repository.save(Document(description=data.description,
                                                               name=data.name,
                                                               file_id=data.file_id,
                                                               source=DocumentSource.UPLOADED))
        return db_doc.id

    async def delete_document_by_id(self, document_id: UUID) -> Document:
        document = await self.get_document_by_id(document_id)
        await self.delete_document(document)
        return document

    async def delete_document(self, document: Document) -> None:
        await self._document_repository.delete(document)

    async def get_embedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        return await self._document_repository.get_embedded(params)

    async def get_unembedded_documents(self, params: PagingParams) -> PagingWrapper[Document]:
        return await self._document_repository.get_unembedded(params)

    async def embed_document(self, store_name: str, doc_id: UUID, chunk_ids: list[str]) -> None:
        db_doc = await self.get_document_by_id(doc_id)
        db_doc.embed_to_vs = store_name
        db_doc.chunks += [DocumentChunk(id=chunk_id) for chunk_id in chunk_ids]
        await self._document_repository.save(db_doc)

    async def unembed_document(self, doc_id: UUID) -> list[str]:
        db_doc = await self.get_document_by_id(doc_id)

        db_chunks = db_doc.chunks
        chunk_ids = [chunk.id for chunk in db_chunks]

        if db_doc.source == DocumentSource.UPLOADED:
            db_doc.embed_to_vs = None
            db_doc.chunks = []
            await self._document_repository.save(db_doc)
        else:
            await self._document_repository.delete(db_doc)

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
                name=d.name,
                source=DocumentSource.EXTERNAL,
                embed_to_vs=store_name,
                chunks=db_chunks)
            docs.append(db_doc)
        await self._document_repository.save_all(docs)

        self._logger.debug(f'Updating external data configuration file with configured status: is_configured = True...')
        with open(file_path, 'w'):  # Clear old content
            pass
        config.is_configured = True  # Mark as configured
        file_path.write_text(config.model_dump_json(indent=2))
