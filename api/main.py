from fastapi import APIRouter

from api.route import agent

api_router = APIRouter()
api_router.include_router(agent.router)