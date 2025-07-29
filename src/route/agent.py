from typing import Literal

from dependency_injector.wiring import inject
from fastapi import APIRouter
from fastapi import status
from fastapi.responses import StreamingResponse

from src.dependency import AgentServiceDepend
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
@inject
async def health_check(service: AgentServiceDepend):
    """Health check endpoint"""
    return service.metadata


@router.post(
    path="/restart",
    status_code=status.HTTP_200_OK,
    description="Restart the agent and return the progressive response stream."
                "A string representing the progress of the restart operation."
                "`{\"status\": \"RESTART\", \"percentage\": 0.0}`."
)
@inject
async def restart(service: AgentServiceDepend):
    async def convert_to_str():
        async for state in service.restart():
            yield str(state)

    return StreamingResponse(convert_to_str(),
                             media_type='text/event-stream',
                             headers={
                                 "Cache-Control": "no-cache",
                                 "Connection": "keep-alive",
                                 "X-Accel-Buffering": "no",
                             })


@router.post("/status", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def set_status(new_status: Literal["ON", "OFF"], service: AgentServiceDepend):
    service.set_status(new_status)
