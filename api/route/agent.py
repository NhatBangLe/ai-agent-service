from typing import Any

from fastapi import APIRouter

router = APIRouter(tags=["test"])


@router.get("/")
def greetings() -> Any:
    return {"message": "Hello world!"}
