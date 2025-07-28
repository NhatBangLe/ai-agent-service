from typing import Literal

from fastapi import APIRouter
from fastapi import status
from fastapi.responses import StreamingResponse

from src.service.interface.agent import AgentMetadata

router = APIRouter(
    prefix="/agent",
    tags=["Agent"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/health", response_model=AgentMetadata, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    from ..main import agent
    return agent.metadata

@router.post(
    path="/restart",
    status_code=status.HTTP_200_OK,
    description="Restart the agent and return the progressive response stream."
                "A string representing the progress of the restart operation."
                "`{\"status\": \"RESTARTING\", \"percentage\": 0.0}`."
)
async def restart():
    from ..main import agent
    async def convert_to_str():
        async for state in agent.restart():
            yield str(state)

    return StreamingResponse(convert_to_str(),
                             media_type='text/event-stream',
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                                 "X-Accel-Buffering": "no",
                             })


@router.post("/status", status_code=status.HTTP_204_NO_CONTENT)
async def set_status(new_status: Literal["ON", "OFF"]):
    from ..main import agent
    agent.set_status(new_status)
