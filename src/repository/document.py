from uuid import UUID

from sqlalchemy import func
from sqlmodel import select

from . import RepositoryImpl
from .interface.document import IDocumentRepository
from ..data.model import Document, DocumentChunk
from ..util import PagingParams, PagingWrapper


# noinspection PyTypeChecker
class DocumentRepositoryImpl(IDocumentRepository, RepositoryImpl):

    async def get_by_id(self, entity_id: UUID) -> Document | None:
        with self._connection.create_session() as session:
            entity = session.get(Document, entity_id)
            return entity

    async def get_all(self) -> list[Document]:
        with self._connection.create_session() as session:
            return list(session.exec(select(Document)).all())

    # noinspection PyComparisonWithNone
    async def get_all_vs_embedded(self) -> list[Document]:
        with self._connection.create_session() as session:
            stmt = select(Document).where(Document.embed_to_vs != None)
            return list(session.exec(stmt).all())

    async def get_embedded(self, params: PagingParams) -> PagingWrapper[Document]:
        with self._connection.create_session() as session:
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
        with self._connection.create_session() as session:
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
        with self._connection.create_session() as session:
            for chunk in chunks:
                session.delete(chunk)
            session.commit()
