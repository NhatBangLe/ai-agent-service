from uuid import UUID

from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse

from ..agent.state import InputState
from ..main import agent
from ..util.main import strict_uuid_parser


def get_thread(thread_id: UUID) -> str:
    """Get thread ID from request"""
    pass


def get_all_messages(thread_id: UUID):
    pass


def create_thread(user_id: UUID):
    pass


def delete_thread(thread_id: UUID):
    pass


router = APIRouter(
    prefix="/threads",
    tags=["Threads"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.post(path="/{user_id}", status_code=status.HTTP_201_CREATED)
async def create(user_id: str):
    """Create a new thread"""
    return create_thread(user_id=strict_uuid_parser(user_id))


@router.get("/{thread_id}", status_code=status.HTTP_200_OK)
async def get_by_id(thread_id: str):
    """Get thread by ID"""
    return get_thread(thread_id=strict_uuid_parser(thread_id))


@router.post(path="/{thread_id}/messages", status_code=status.HTTP_200_OK)
async def append_message(thread_id: str, message: InputState):
    """Add a message and stream response"""
    return StreamingResponse(
        agent.stream(
            input_msg=message,
            config={
                "configurable": {"thread_id": thread_id}
            }
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get(path="/{thread_id}/messages", status_code=status.HTTP_200_OK)
async def get_messages(thread_id: str):
    """Get all messages in a thread"""
    return get_all_messages(thread_id=strict_uuid_parser(thread_id))


@router.delete(path="/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(thread_id: str) -> None:
    """Delete a thread (assistant-ui compatible)"""
    delete_thread(thread_id=strict_uuid_parser(thread_id))
