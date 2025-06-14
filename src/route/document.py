import datetime
import os.path
import typing
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, status
from sqlmodel import Session, select

from ..data.base_model import DocumentSource
from ..data.dto import DocumentPublic
from ..data.model import Document, DocumentChunk
from ..dependency import SessionDep, DownloadGeneratorDep, PagingParams, PagingQuery
from ..util.error import NotFoundError
from ..util.main import SecureDownloadGenerator, FileInformation
from ..util.function import strict_uuid_parser
from ..util.constant import DEFAULT_TIMEZONE

DEFAULT_SAVE_DIRECTORY = "/resource"


def get_save_document_directory():
    return os.getenv("SAVE_DOCUMENT_DIRECTORY", DEFAULT_SAVE_DIRECTORY)


def get_document(doc_id: UUID, session: Session) -> Document:
    db_doc = session.get(Document, doc_id)
    if db_doc is None:
        raise NotFoundError(f'Document with id {doc_id} not found.')
    return typing.cast(Document, db_doc)


def get_document_download_token(doc_id: UUID, session: Session, generator: SecureDownloadGenerator) -> str:
    db_doc = get_document(doc_id, session)
    if db_doc.save_path is None:
        raise NotFoundError(f'Document {db_doc.name} has not been saved, its source is {db_doc.source.name}.')

    data: FileInformation = {
        "name": db_doc.name,
        "mime_type": db_doc.mime_type,
        "path": db_doc.save_path,
    }
    return generator.generate_token(data=data)


# noinspection PyTypeChecker,PyComparisonWithNone
def get_embedded_documents(params: PagingParams, session: Session) -> list[Document]:
    statement = (select(Document)
                 .distinct()
                 .join(DocumentChunk, DocumentChunk.document_id == Document.id)
                 .offset((params.offset * params.limit))
                 .limit(params.limit)
                 .order_by(Document.created_at))
    results = session.exec(statement)
    return list(results.all())


# noinspection PyTypeChecker,PyComparisonWithNone
def get_unembedded_documents(params: PagingParams, session: Session) -> list[Document]:
    statement = (select(Document)
                 .join(DocumentChunk, DocumentChunk.document_id == Document.id, isouter=True)
                 .where(DocumentChunk.document_id == None)
                 .offset((params.offset * params.limit))
                 .limit(params.limit)
                 .order_by(Document.created_at))
    results = session.exec(statement)
    return list(results.all())


async def save_document(file: UploadFile, session: Session) -> UUID:
    file_bytes = await file.read()
    doc_id = uuid4()
    save_path = os.path.join(get_save_document_directory(), str(doc_id))
    Path(save_path).write_bytes(file_bytes)

    db_doc = Document(
        id=doc_id,
        created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
        name=file.filename,
        mime_type=file.content_type,
        save_path=save_path,
        source=DocumentSource.UPLOADED
    )
    session.add(db_doc)
    session.commit()

    return doc_id


async def embed_document(store_name: str, doc_id: UUID, session: Session):
    db_doc = get_document(doc_id, session)

    from ..main import get_agent
    agent = get_agent()
    try:
        await agent.embed_document(
            store_name=store_name,
            file_info={
                "name": db_doc.name,
                "path": db_doc.save_path,
                "mime_type": db_doc.mime_type,
            })

        db_doc.embed_to_vs = store_name
        session.add(db_doc)
        session.commit()
    except NotImplementedError:
        raise NotFoundError(f'Do not have vector store with name {store_name}')


def delete_document(doc_id: UUID, session: Session):
    db_doc = get_document(doc_id, session)
    session.delete(db_doc)
    session.commit()


router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/{document_id}/token", status_code=status.HTTP_200_OK)
async def get_download_token(document_id: str, session: SessionDep,
                             generator: DownloadGeneratorDep) -> str:
    doc_uuid = strict_uuid_parser(document_id)
    return get_document_download_token(doc_id=doc_uuid, session=session, generator=generator)


@router.get("/{document_id}/info", response_model=DocumentPublic, status_code=status.HTTP_200_OK)
async def get_information(document_id: str, session: SessionDep):
    return get_document(doc_id=strict_uuid_parser(document_id), session=session)


@router.get("/embedded", response_model=list[DocumentPublic], status_code=status.HTTP_200_OK)
async def get_embedded(params: PagingQuery, session: SessionDep):
    return get_embedded_documents(params=params, session=session)


@router.get("/unembedded", response_model=list[DocumentPublic], status_code=status.HTTP_200_OK)
async def get_unembedded(params: PagingQuery, session: SessionDep):
    return get_unembedded_documents(params=params, session=session)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload(file: UploadFile, session: SessionDep) -> str:
    uploaded_document_id = await save_document(file=file, session=session)
    return str(uploaded_document_id)


@router.post("/{store_name}/embed/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def embed(store_name: str, document_id: str, session: SessionDep) -> None:
    doc_uuid = strict_uuid_parser(document_id)
    await embed_document(store_name=store_name, doc_id=doc_uuid, session=session)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(document_id: str, session: SessionDep) -> None:
    delete_document(doc_id=strict_uuid_parser(document_id), session=session)
