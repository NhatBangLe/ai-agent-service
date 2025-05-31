import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.agent.agent import Agent
from src.api.image import router as image_router
from src.api.label import router as label_router
from src.config.main import get_config_folder_path
from src.data.database import DatabaseManager
from src.error import NotFoundError, InvalidArgumentError


# Set up logging.
def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO")
    matches = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
    }
    logging.basicConfig(level=matches[level])


# Initialize
setup_logging()
db_manager = DatabaseManager()
agent = Agent()


# noinspection PyUnusedLocal
@asynccontextmanager
async def lifespan(api: FastAPI):
    # Create database tables.
    db_manager.create_db_and_tables()

    # Insert predefined output classes to the database.
    image_recognizer_config = agent.configurer.config.image_recognizer
    if image_recognizer_config is not None:
        config_file_path = os.path.join(get_config_folder_path(), image_recognizer_config.output_config_path)
        db_manager.insert_predefined_output_classes(config_file_path)

    # Initialize the agent.
    agent.build_graph()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=image_router)
app.include_router(router=label_router)


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy"
    }


# noinspection PyUnusedLocal
@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request: Request, exc: NotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": exc.reason},
    )


# noinspection PyUnusedLocal
@app.exception_handler(InvalidArgumentError)
async def invalid_argument_exception_handler(request: Request, exc: InvalidArgumentError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"message": exc.reason},
    )
