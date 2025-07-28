import logging
from pathlib import Path

from .interface.document import IDocumentService
from .interface.file import IFileService
from ..config.model.data import ExternalDocumentConfiguration
from ..data.base_model import DocumentSource
from ..data.model import DocumentChunk, Document
from ..repository.interface.document import IDocumentRepository
from ..util.constant import SUPPORTED_DOCUMENT_TYPE_DICT
from ..util.error import NotFoundError
from ..util.function import shrink_file_name


class DocumentServiceImpl(IDocumentService):
    _document_repository: IDocumentRepository
    _file_service: IFileService
    _logger = logging.getLogger(__name__)

    def __init__(self, document_repository: IDocumentRepository,
                 file_service: IFileService):
        super().__init__()
        self._document_repository = document_repository
        self._file_service = file_service

    async def get_document_by_id(self, document_id):
        doc = await self._document_repository.get_by_id(entity_id=document_id)
        if doc is None:
            raise NotFoundError(f'Document with id {document_id} not found.')
        return doc

    async def save_document(self, data):
        file_name = shrink_file_name(150, data.name, SUPPORTED_DOCUMENT_TYPE_DICT[data.mime_type])

        # Save the uploaded file by using the file service
        save_file = IFileService.SaveFile(name=file_name, mime_type=data.mime_type, data=data.data)
        file_metadata = await self._file_service.save_file(save_file)
        db_doc = await self._document_repository.save(Document(description=data.description,
                                                               name=data.name,
                                                               file_id=file_metadata.id,
                                                               source=DocumentSource.UPLOADED))
        return db_doc

    async def delete_document_by_id(self, document_id):
        with await self._document_repository.get_session() as session:
            document: Document | None = session.get(Document, document_id)
            if document is None:
                raise NotFoundError(f'Document with id {document_id} not found.')
            file = document.file
            session.delete(document)
            session.commit()
        if file.thread_id is None:
            await self._file_service.delete_file_by_id(file.id)
        return document

    async def delete_document(self, document):
        await self.delete_document_by_id(document.id)

    async def get_embedded_documents(self, params):
        return await self._document_repository.get_embedded(params)

    async def get_unembedded_documents(self, params):
        return await self._document_repository.get_unembedded(params)

    async def embed_document(self, store_name, doc_id, chunk_ids):
        db_doc = await self.get_document_by_id(doc_id)
        db_doc.embed_to_vs = store_name
        db_doc.chunks += [DocumentChunk(id=chunk_id) for chunk_id in chunk_ids]
        await self._document_repository.save(db_doc)

    async def unembed_document(self, doc_id):
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

    async def insert_external_document(self, store_name, file_path):
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
