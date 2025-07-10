from typing import Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, UploadFile, status, File, Form

from ..data.base_model import DocumentSource
from ..data.dto import DocumentPublic
from ..data.model import Document
from ..dependency import DownloadGeneratorDepend, PagingQuery, DocumentServiceDepend
from ..util import FileInformation, PagingWrapper
from ..util.constant import SUPPORTED_DOCUMENT_TYPE_DICT
from ..util.error import NotFoundError, InvalidArgumentError
from ..util.function import strict_uuid_parser


def to_doc_public(db_doc: Document):
    return DocumentPublic(
        id=db_doc.id,
        name=db_doc.name,
        description=db_doc.description,
        created_at=db_doc.created_at,
        mime_type=db_doc.mime_type,
        source=db_doc.source,
        embedded_to_vs=db_doc.embed_to_vs,
        embedded_to_bm25=db_doc.embed_bm25
    )


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/{document_id}/token", status_code=status.HTTP_200_OK)
@inject
async def get_download_token(document_id: str,
                             service: DocumentServiceDepend,
                             generator: DownloadGeneratorDepend) -> str:
    doc_uuid = strict_uuid_parser(document_id)
    db_doc = await service.get_document_by_id(doc_uuid)
    if db_doc.source == DocumentSource.EXTERNAL:
        raise InvalidArgumentError(f'Cannot download document because the document is from external source.')
    if db_doc.save_path is None:
        raise NotFoundError(f'Document {db_doc.name} has not been saved, its source is {db_doc.source.name}.')

    data: FileInformation = {
        "name": db_doc.name,
        "mime_type": str(db_doc.mime_type),  # mime_type is not None if source != DocumentSource.EXTERNAL
        "path": db_doc.save_path,
    }
    return generator.generate_token(data=data)


@router.get("/{document_id}/info", response_model=DocumentPublic, status_code=status.HTTP_200_OK)
@inject
async def get_information(document_id: str, service: DocumentServiceDepend):
    doc_uuid = strict_uuid_parser(document_id)
    doc = await service.get_document_by_id(doc_uuid)
    return to_doc_public(doc)


@router.get("/embedded", response_model=PagingWrapper[DocumentPublic], status_code=status.HTTP_200_OK)
@inject
async def get_embedded(params: PagingQuery, service: DocumentServiceDepend):
    db_paging = await service.get_embedded_documents(params)
    return PagingWrapper.convert_content_type(db_paging, to_doc_public)


@router.get("/unembedded", response_model=PagingWrapper[DocumentPublic], status_code=status.HTTP_200_OK)
@inject
async def get_unembedded(params: PagingQuery, service: DocumentServiceDepend):
    db_paging = await service.get_unembedded_documents(params)
    return PagingWrapper.convert_content_type(db_paging, to_doc_public)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
@inject
async def upload(file: Annotated[UploadFile, File()],
                 description: Annotated[str | None, Form(max_length=255)],
                 service: DocumentServiceDepend) -> str:
    mime_type = file.content_type
    if mime_type not in SUPPORTED_DOCUMENT_TYPE_DICT:
        raise InvalidArgumentError(f'Unsupported MIME type: {mime_type}.')
    uploaded_document_id = await service.save_document(file, description)
    return str(uploaded_document_id)


@router.post("/{store_name}/embed/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def embed(store_name: str, document_id: str, service: DocumentServiceDepend) -> None:
    doc_uuid = strict_uuid_parser(document_id)
    db_doc = service.get_document_by_id(doc_uuid)

    from ..main import get_agent
    agent = get_agent()
    try:
        chunk_ids = await agent.embed_document(
            store_name=store_name,
            file_info={
                "name": db_doc.name,
                "path": db_doc.save_path,
                "mime_type": db_doc.mime_type,
            })
        await service.embed_document(store_name=store_name, doc_id=doc_uuid, chunk_ids=chunk_ids)
    except ValueError:
        raise NotFoundError(f'Do not have vector store with name {store_name}')


@router.delete("/{document_id}/unembed", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def unembed(document_id: str, service: DocumentServiceDepend) -> None:
    doc_uuid = strict_uuid_parser(document_id)
    db_doc = await service.get_document_by_id(doc_uuid)
    store_name = db_doc.embed_to_vs
    if store_name is None:
        raise NotFoundError(f'Document {db_doc.name} has not been embedded to vector store.')

    chunk_ids = await service.unembed_document(store_name=store_name, doc_id=doc_uuid)

    try:
        from ..main import get_agent
        agent = get_agent()
        await agent.unembed_document(store_name=store_name, chunk_ids=chunk_ids)
    except ValueError:
        raise NotFoundError(f'Do not have vector store with name {store_name}')


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(document_id: str, service: DocumentServiceDepend) -> None:
    doc_uuid = strict_uuid_parser(document_id)
    await service.delete_document_by_id(doc_uuid)
