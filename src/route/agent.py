import asyncio
from typing import Literal

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import StreamingResponse

router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    from ..main import agent
    return agent.status


@router.post("/bm25/sync", status_code=status.HTTP_200_OK)
async def sync_bm25():
    from ..main import agent
    asyncio.create_task(coro=agent.sync_bm25(), name="sync_bm25")


@router.post(
    path="/restart",
    status_code=status.HTTP_200_OK,
    description="Restart the agent and return the progressive response stream."
                "A string representing the progress of the restart operation."
                "`{\"status\": \"RESTARTING\", \"percentage\": 0.0}`, use a new line character to separate lines."
)
async def restart():
    from ..main import agent
    def convert_to_str():
        for state in agent.restart():
            yield f'{state}\n'

    return StreamingResponse(convert_to_str(),
                             media_type='text/event-stream',
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                                 "X-Accel-Buffering": "no",
                             })


@router.post("/status", status_code=status.HTTP_200_OK)
async def set_status(new_status: Literal["ON", "OFF"]):
    from ..main import agent
    agent.status = new_status
