import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse

from src.agent.agent import Agent
from src.dependency import DownloadGeneratorDep
from src.route.image import router as image_router
from src.route.label import router as label_router
from src.route.export import router as export_router
from src.route.document import router as document_router
from src.config.main import get_config_folder_path, AgentConfigurer
from src.data.database import insert_predefined_output_classes, create_db_and_tables
from src.util.error import NotFoundError, InvalidArgumentError


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
load_dotenv()
configurer = AgentConfigurer()
agent = Agent(configurer=configurer)


# noinspection PyUnusedLocal
@asynccontextmanager
async def lifespan(api: FastAPI):
    # Initialize the agent.
    agent.configure()
    agent.build_graph()

    # Create database tables.
    create_db_and_tables()

    # Insert predefined output classes to the database.
    image_recognizer_config = agent.configurer.config.image_recognizer
    if image_recognizer_config is not None:
        config_file_path = os.path.join(get_config_folder_path(), image_recognizer_config.output_config_path)
        insert_predefined_output_classes(config_file_path)

    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router=image_router)
app.include_router(router=label_router)
app.include_router(router=export_router)
app.include_router(router=document_router)


# Global routes
@app.get("/download", tags=["Download File"], status_code=status.HTTP_200_OK)
async def download(token: str, generator: DownloadGeneratorDep):
    file = generator.verify_token(token)
    print(f'Downloading file: {file["name"]}')
    return FileResponse(
        path=file["path"],
        media_type=file["mime_type"],
        filename=file["name"]
    )


@app.get(
    path="/restart",
    tags=["Agent"],
    status_code=status.HTTP_200_OK,
    description="Restart the agent and return the progressive response stream."
                "A string representing the progress of the restart operation."
                "`{\"status\": \"RESTARTING\", \"percentage\": 0.0}`, use a new line character to separate lines."
)
async def restart():
    return StreamingResponse(agent.restart(), media_type='text/event-stream')


@app.get("/health", tags=["Agent"], status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {
        "status": agent.status
    }


# Exception handlers
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
