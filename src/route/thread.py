import math
from typing import cast, Literal, Annotated
from uuid import UUID

from dependency_injector.wiring import inject
from fastapi import APIRouter, status, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, BaseMessage, AIMessageChunk, AIMessage

from ..agent import Attachment, State
from ..data.dto import InputMessage, OutputMessage, ThreadPublic, ThreadCreate, ThreadUpdate
from ..dependency import PagingQuery, ThreadServiceDepend, ImageServiceDepend, FileServiceDepend
from ..util import PagingWrapper, PagingParams
from ..util.function import strict_uuid_parser


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


router = APIRouter(
    prefix="/threads",
    tags=["Threads"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get(path="/{user_id}/all", response_model=PagingWrapper[ThreadPublic], status_code=status.HTTP_200_OK)
@inject
async def get_all_threads(user_id: str, params: PagingQuery, service: ThreadServiceDepend):
    return await service.get_all_by_user_id(user_id=strict_uuid_parser(user_id), params=params)


@router.get("/{thread_id}", response_model=ThreadPublic, status_code=status.HTTP_200_OK)
@inject
async def get_by_id(thread_id: str, service: ThreadServiceDepend):
    return await service.get_thread_by_id(strict_uuid_parser(thread_id))


@router.get(path="/{thread_id}/messages", response_model=PagingWrapper[OutputMessage], status_code=status.HTTP_200_OK)
@inject
async def get_all_messages(thread_id: str, params: PagingQuery):
    """Get all messages in a thread"""
    return await get_all_messages_from_thread(thread_id=strict_uuid_parser(thread_id), params=params)


@router.post(path="/{user_id}/create", status_code=status.HTTP_201_CREATED)
@inject
async def create_thread(user_id: str, data: ThreadCreate, service: ThreadServiceDepend) -> str:
    new_id = await service.create_thread(user_id=strict_uuid_parser(user_id), data=data)
    return str(new_id)


@router.put(path="/{thread_id}/update", status_code=status.HTTP_201_CREATED)
@inject
async def update_thread(thread_id: str, data: ThreadUpdate, service: ThreadServiceDepend) -> None:
    await service.update_thread(thread_id=strict_uuid_parser(thread_id), data=data)


@router.post(path="/{thread_id}/messages", status_code=status.HTTP_200_OK)
@inject
async def append_message(thread_id: str,
                         input_msg: InputMessage,
                         stream_mode: Annotated[Literal["values", "updates", "messages"], Query()],
                         image_service: ImageServiceDepend,
                         file_service: FileServiceDepend):
    """Add a message and stream response"""

    async def get_chunk():
        from ..main import get_agent

        attachment: Attachment | None = None
        if input_msg.attachment is not None:
            db_image = await image_service.get_image_by_id(input_msg.attachment.id)
            file = await file_service.get_file_by_id(db_image.file_id)
            if db_image is None or file is None:
                raise ValueError("Attachment not found.")
            attachment = Attachment(id=str(db_image.id),
                                    name=file.name,
                                    mime_type=file.mime_type,
                                    path=file.path)

        input_state: State = {
            "messages": [HumanMessage(input_msg.content)],
            "attachment": attachment
        }
        agent = get_agent()

        async for state in agent.astream(
                input_state=input_state,
                stream_mode=stream_mode,
                config={
                    "configurable": {"thread_id": thread_id}
                }
        ):
            if stream_mode == "values":
                msgs: list[BaseMessage] = state["messages"]
                yield str([msg.model_dump_json() for msg in msgs])
            elif stream_mode == "updates":
                yield str([value for _, value in state.items()])
            elif stream_mode == "messages":
                chunk, langgraph_metadata = state
                yield chunk.model_dump_json()

    return StreamingResponse(
        get_chunk(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.delete(path="/{thread_id}/delete", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(thread_id: str, service: ThreadServiceDepend) -> None:
    await service.delete_thread_by_id(strict_uuid_parser(thread_id))
