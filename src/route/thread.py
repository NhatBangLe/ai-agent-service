import asyncio
import math
from typing import Literal, Annotated
from uuid import UUID

import aiohttp
from dependency_injector.wiring import inject
from fastapi import APIRouter, status, Query, UploadFile, Request
from fastapi.responses import StreamingResponse, FileResponse
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from ..service.interface.agent import Attachment, IAgentService
from ..data.dto import InputMessage, ThreadPublic, ThreadCreate, ThreadUpdate, ImageCreate, AttachmentPublic
from ..dependency import PagingQuery, ThreadServiceDepend, FileServiceDepend, ImageServiceDepend, AgentServiceDepend
from ..service.interface.file import IFileService
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError, InvalidArgumentError
from ..util.function import strict_uuid_parser, is_web_path, get_cache_dir_path


async def get_all_messages_from_thread(thread_id: UUID, params: PagingParams,
                                       agent_service: IAgentService) -> PagingWrapper:
    config: RunnableConfig = {"configurable": {"thread_id": str(thread_id)}}
    states = await agent_service.get_state_history(config, limit=1)
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

    messages: list[BaseMessage] = states[0].values["messages"][::-1]
    messages_len = len(messages)

    total_pages = math.ceil(messages_len / params.limit)
    start_idx = params.offset * params.limit
    end_idx = min(start_idx + params.limit, messages_len)
    return PagingWrapper(
        content=messages[start_idx:end_idx],
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
    return await service.get_all_threads_by_user_id(user_id=strict_uuid_parser(user_id), params=params)


@router.get("/{thread_id}", response_model=ThreadPublic, status_code=status.HTTP_200_OK)
@inject
async def get_by_id(thread_id: str, service: ThreadServiceDepend):
    return await service.get_thread_by_id(strict_uuid_parser(thread_id))


@router.get(path="/{thread_id}/messages", response_model=PagingWrapper, status_code=status.HTTP_200_OK)
@inject
async def get_all_messages(thread_id: str, params: PagingQuery, agent_service: AgentServiceDepend):
    """Get all messages in a thread"""
    return await get_all_messages_from_thread(thread_id=strict_uuid_parser(thread_id),
                                              params=params, agent_service=agent_service)


@router.post(path="/{user_id}/create", status_code=status.HTTP_201_CREATED)
@inject
async def create_thread(user_id: str, data: ThreadCreate, service: ThreadServiceDepend) -> str:
    thread = await service.create_thread(user_id=strict_uuid_parser(user_id), data=data)
    return str(thread.id)


@router.put(path="/{thread_id}/update", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def update_thread(thread_id: str, data: ThreadUpdate, service: ThreadServiceDepend) -> None:
    await service.update_thread(thread_id=strict_uuid_parser(thread_id), data=data)


@router.post(path="/{thread_id}/messages", status_code=status.HTTP_200_OK)
@inject
async def append_message(thread_id: str,
                         input_msg: InputMessage,
                         stream_mode: Annotated[Literal["values", "updates", "messages"], Query()],
                         file_service: FileServiceDepend,
                         thread_service: ThreadServiceDepend,
                         agent_service: AgentServiceDepend):
    """Add a message and stream response"""
    if input_msg.attachment_id is None and len(input_msg.content.strip()) == 0:
        raise InvalidArgumentError("Attachment and content cannot be empty at the same time.")

    await thread_service.get_thread_by_id(strict_uuid_parser(thread_id))
    attachment: Attachment | None = None
    attachment_id = input_msg.attachment_id
    if attachment_id is not None:
        file = await file_service.get_metadata_by_id(strict_uuid_parser(attachment_id))
        if file is None:
            raise NotFoundError("Attachment not found.")
        attachment = Attachment(id=attachment_id,
                                name=file.name,
                                mime_type=file.mime_type,
                                path=file.path)

    async def get_chunk():
        async for state in agent_service.astream(
                input_state={
                    "messages": [HumanMessage(content=input_msg.content,
                                              additional_kwargs={"attachment": attachment})],
                },
                stream_mode=stream_mode,
                config={
                    "configurable": {"thread_id": thread_id},
                    # "recursion_limit": 5,
                }
        ):
            if stream_mode == "values":
                msgs: list[BaseMessage] = state["messages"]
                yield str([msg.model_dump_json() for msg in msgs])
            elif stream_mode == "updates":
                yield str([value for _, value in state.items()])
            elif stream_mode == "messages":
                chunk, langgraph_metadata = state
                print(chunk)
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


@router.get("/attachment/{attachment_id}/metadata",
            response_model=AttachmentPublic,
            status_code=status.HTTP_200_OK)
@inject
async def get_attachment_metadata(attachment_id: str, request: Request, service: FileServiceDepend):
    file = await service.get_metadata_by_id(strict_uuid_parser(attachment_id))
    if file is None:
        raise NotFoundError(f"Attachment with id {attachment_id} not found.")

    return AttachmentPublic(
        id=attachment_id,
        name=file.name,
        mime_type=file.mime_type,
        url=str(request.url).replace('/metadata', ''))


@router.get("/attachment/{attachment_id}", status_code=status.HTTP_200_OK)
@inject
async def get_attachment(attachment_id: str, service: FileServiceDepend):
    file = await service.get_metadata_by_id(strict_uuid_parser(attachment_id))
    if file is None:
        raise NotFoundError(f"Attachment with id {attachment_id} not found.")
    path = file.path
    if is_web_path(path):
        cache_dir = get_cache_dir_path()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir.joinpath(attachment_id)
        if not cached_file.exists():
            async with aiohttp.ClientSession() as session:
                response = await session.get(path)
                cached_file.write_bytes(await response.read())
        path = str(cached_file)
    return FileResponse(path=path, media_type=file.mime_type, filename=file.name)


@router.post(path="/attachment/{thread_id}/upload", status_code=status.HTTP_200_OK)
@inject
async def upload_attachment(thread_id: str, file: UploadFile,
                            service: ThreadServiceDepend,
                            image_service: ImageServiceDepend,
                            agent_service: AgentServiceDepend,
                            file_service: FileServiceDepend) -> str:
    file_bytes = await file.read()
    if "image" in file.content_type:
        image = await image_service.save_image(ImageCreate(name=file.filename,
                                                           mime_type=file.content_type,
                                                           data=file_bytes))
        attachment_id = image.file_id

        img_recognizer = agent_service.configurer.image_recognizer
        if img_recognizer is not None:
            from .image import predict_labels
            asyncio.create_task(predict_labels(img_recognizer, file_bytes, image.id, image_service))
    else:
        file = await file_service.save_file(IFileService.SaveFile(name=file.filename,
                                                                  mime_type=file.content_type,
                                                                  data=file_bytes))
        attachment_id = file.id

    await service.add_attachments(thread_id, [attachment_id])
    return str(attachment_id)


@router.delete(path="/attachment/{attachment_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete_attachment(attachment_id: str, service: ThreadServiceDepend) -> None:
    await service.delete_attachment_by_id(strict_uuid_parser(attachment_id))
