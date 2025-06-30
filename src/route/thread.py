import datetime
import math
from typing import cast, Literal
from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, BaseMessage, AIMessageChunk, AIMessage
from sqlalchemy import func
from sqlmodel import Session, select

from ..agent import InputState, Attachment
from ..data.dto import InputMessage, OutputMessage, ThreadPublic, ThreadCreate, ThreadUpdate
from ..data.model import Thread, User, Image
from ..dependency import SessionDep, PagingQuery
from ..util import PagingWrapper, PagingParams
from ..util.constant import DEFAULT_TIMEZONE
from ..util.function import strict_uuid_parser


def get_all_threads_by_user_id(user_id: UUID, params: PagingParams, session: Session):
    count_statement = (select(func.count())
                       .where(Thread.user_id == user_id))
    execute_statement = (select(Thread)
                         .where(Thread.user_id == user_id)
                         .offset(params.offset)
                         .limit(params.limit)
                         .order_by(Thread.created_at))
    return PagingWrapper.get_paging(params, count_statement, execute_statement, session)


def get_thread(thread_id: UUID, session: Session) -> Thread:
    db_thread = session.get(Thread, thread_id)
    return cast(Thread, db_thread)


async def get_all_messages_from_thread(thread_id: UUID, params: PagingParams) -> PagingWrapper[OutputMessage]:
    from ..main import get_agent
    agent = get_agent()
    config = {"configurable": {"thread_id": str(thread_id)}}
    states = await agent.get_state_history(config, limit=1)
    if len(states) < 1:
        return PagingWrapper(
            content=[],
            first=True,
            last=True,
            page_number=params.offset,
            page_size=params.limit,
            total_pages=0,
            total_elements=0
        )

    def convert_to_output_message(message: BaseMessage):
        role: Literal["Human", "AI"] = "Human"
        if isinstance(message, AIMessage):
            role = "AI"
        return OutputMessage(
            id=message.id,
            content=message.content,
            role=role
        )

    results: list[OutputMessage] = []
    messages: list[BaseMessage] = states[0].values["messages"]
    messages_len = len(messages)
    i = 0
    while i < messages_len:
        if isinstance(messages[i], AIMessageChunk):
            ai_message = cast(AIMessageChunk, messages[i])
            j = i + 1
            while isinstance(messages[j], AIMessageChunk) and j < messages_len:
                ai_message += cast(AIMessageChunk, messages[j])
                j += 1
            results.append(convert_to_output_message(ai_message))
            i = j - 1
        elif isinstance(messages[i], (HumanMessage, AIMessage)):
            results.append(convert_to_output_message(messages[i]))
        i += 1

    total_pages = math.ceil(messages_len / params.limit)
    return PagingWrapper(
        content=results[messages_len - params.limit:],
        first=params.offset == 0,
        last=params.offset == total_pages - 1,
        page_number=params.offset,
        page_size=params.limit,
        total_pages=total_pages,
        total_elements=messages_len
    )


def create_thread(user_id: UUID, data: ThreadCreate, session: Session) -> UUID:
    db_user = session.get(User, user_id)
    if db_user is None:
        db_user = User(id=user_id)
    db_thread = Thread(
        title=data.title,
        created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
        user=db_user
    )
    session.add(db_thread)
    session.commit()

    return db_thread.id


def update_thread(thread_id: UUID, data: ThreadUpdate, session: Session):
    db_thread = get_thread(thread_id, session)
    db_thread.title = data.title
    session.add(db_thread)
    session.commit()


def delete_thread(thread_id: UUID, session: Session):
    from ..main import get_agent
    agent = get_agent()
    agent.delete_thread(thread_id)

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
    return await get_all_messages_from_thread(thread_id=strict_uuid_parser(thread_id), params=params)


@router.post(path="/{user_id}", status_code=status.HTTP_201_CREATED)
async def create(user_id: str, data: ThreadCreate, session: SessionDep) -> str:
    """Create a new thread"""
    new_id = create_thread(user_id=strict_uuid_parser(user_id), data=data, session=session)
    return str(new_id)


@router.put(path="/{thread_id}", status_code=status.HTTP_201_CREATED)
async def update(thread_id: str, data: ThreadUpdate, session: SessionDep) -> None:
    """Update a new thread"""
    update_thread(thread_id=strict_uuid_parser(thread_id), data=data, session=session)


@router.post(path="/{thread_id}/messages", status_code=status.HTTP_200_OK)
async def append_message(thread_id: str, input_msg: InputMessage, session: SessionDep):
    """Add a message and stream response"""

    # noinspection PyUnresolvedReferences
    async def get_chunk():
        from ..main import get_agent

        attachments: list[Attachment] | None = None
        if input_msg.attachments is not None and len(input_msg.attachments) != 0:
            img_ids = [strict_uuid_parser(atm.id) for atm in input_msg.attachments]
            db_images: list[Image] = list(session.exec(select(Image).where(Image.id.in_(img_ids))).all())
            attachments = [Attachment(id=str(img.id),
                                      name=img.name,
                                      mime_type=img.mime_type,
                                      save_path=img.save_path) for img in db_images]

        input_state: InputState = {
            "messages": [HumanMessage(input_msg.content)],
            "attachments": attachments
        }
        agent = get_agent()

        async for chunk, metadata in agent.astream(
                input_msg=input_state,
                stream_mode="messages",
                config={
                    "configurable": {"thread_id": thread_id}
                }
        ):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(
        get_chunk(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.delete(path="/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(thread_id: str, session: SessionDep) -> None:
    """Delete a thread"""
    delete_thread(thread_id=strict_uuid_parser(thread_id), session=session)
