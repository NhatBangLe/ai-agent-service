import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, status
from sqlalchemy import func
from sqlmodel import Session, select

from ..data.dto import PagingWrapper, OutputMessage, ThreadPublic, ThreadCreate
from ..data.model import Thread
from ..dependency import SessionDep, PagingParams, PagingQuery
from ..util.constant import DEFAULT_TIMEZONE
from ..util.function import strict_uuid_parser, get_paging


def get_all_threads_by_user_id(user_id: UUID, params: PagingParams, session: Session):
    count_statement = (select(func.count())
                       .where(Thread.user_id == user_id))
    execute_statement = (select(Thread)
                         .where(Thread.user_id == user_id)
                         .offset(params.offset)
                         .limit(params.limit)
                         .order_by(Thread.created_at))
    return get_paging(params, count_statement, execute_statement, session)


def get_thread(thread_id: UUID, session: Session) -> Thread:
    db_thread = session.get(Thread, thread_id)
    return cast(Thread, db_thread)


def get_all_messages_from_thread(thread_id: UUID, params: PagingParams) -> PagingWrapper[OutputMessage]:
    return PagingWrapper(
        content=[],
        first=True,
        last=True,
        page_number=params.offset,
        page_size=params.limit,
        total_pages=0,
        total_elements=0
    )


def create_thread(user_id: UUID, data: ThreadCreate, session: Session) -> UUID:
    db_thread = Thread(
        title=data.title,
        created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
        user_id=user_id
    )
    session.add(db_thread)
    session.commit()

    return db_thread.id


def delete_thread(thread_id: UUID, session: Session):
    db_thread = session.get(Thread, thread_id)
    session.delete(db_thread)
    session.commit()


router = APIRouter(
    prefix="/threads",
    tags=["Threads"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get(path="/{user_id}/all", response_model=PagingWrapper[ThreadPublic], status_code=status.HTTP_200_OK)
async def get_all_threads(user_id: str, params: PagingQuery, session: SessionDep):
    """Get all messages in a thread"""
    return get_all_threads_by_user_id(user_id=strict_uuid_parser(user_id), params=params, session=session)


@router.get("/{thread_id}", response_model=ThreadPublic, status_code=status.HTTP_200_OK)
async def get_by_id(thread_id: str, session: SessionDep):
    """Get thread by ID"""
    return get_thread(thread_id=strict_uuid_parser(thread_id), session=session)


@router.get(path="/{thread_id}/messages", response_model=PagingWrapper[OutputMessage], status_code=status.HTTP_200_OK)
async def get_all_messages(thread_id: str, params: PagingQuery):
    """Get all messages in a thread"""
    return get_all_messages_from_thread(thread_id=strict_uuid_parser(thread_id), params=params)


@router.post(path="/{user_id}", status_code=status.HTTP_201_CREATED)
async def create(user_id: str, data: ThreadCreate, session: SessionDep) -> str:
    """Create a new thread"""
    new_id = create_thread(user_id=strict_uuid_parser(user_id), data=data, session=session)
    return str(new_id)


@router.delete(path="/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(thread_id: str, session: SessionDep) -> None:
    """Delete a thread"""
    delete_thread(thread_id=strict_uuid_parser(thread_id), session=session)
