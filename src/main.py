import asyncio
import logging
import os
import platform
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from src.agent.agent import Agent
from src.config.configurer.agent import AgentConfigurer
from src.config.model.retriever.vector_store import VectorStoreConfiguration
from src.data.database import insert_predefined_output_classes, create_db_and_tables, insert_external_data
from src.dependency import DownloadGeneratorDep
from src.route.agent import router as agent_router
from src.route.document import router as document_router
from src.route.export import router as export_router
from src.route.image import router as image_router
from src.route.label import router as label_router
from src.route.thread import router as thread_router
from src.util.error import NotFoundError, InvalidArgumentError
from src.util.function import get_config_folder_path


# Set up logging.
def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO")
    matches = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
    }

    pattern = (
        "%(asctime)s - %(levelname)s - %(name)s - "
        "%(filename)s:%(lineno)d - %(message)s"
    )
    logging.basicConfig(level=matches[level], format=pattern)


def setup_event_loop():
    if 'Windows' in platform.system():
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )


# Initialize
load_dotenv()
setup_event_loop()
setup_logging()
configurer = AgentConfigurer()
agent = Agent(configurer=configurer)


def get_agent():
    agent.check_graph_available()
    return agent


# noinspection PyUnusedLocal
@asynccontextmanager
async def lifespan(api: FastAPI):
    # Create database tables.
    create_db_and_tables()

    # Initialize the agent.
    await agent.configure()
    agent.build_graph()

    agent_config = agent.configurer.config
    # Insert predefined output classes to the database.
    if agent_config.image_recognizer is not None:
        recognizer_output_config_path = agent_config.image_recognizer.output_config_path
        config_file_path = os.path.join(get_config_folder_path(), recognizer_output_config_path)
        insert_predefined_output_classes(str(config_file_path))

    # Insert external data from vector stores
    retriever_configs = agent_config.retrievers
    if retriever_configs is not None:
        vs_configs: list = list(
            filter(lambda config: isinstance(config, VectorStoreConfiguration), retriever_configs))
        for c in vs_configs:
            path = c.external_data_config_path
            if path is not None:
                config_file_path = os.path.join(get_config_folder_path(), path)
                insert_external_data(str(config_file_path))

    yield

    await agent.shutdown()


app = FastAPI(lifespan=lifespan)
# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router=agent_router)
app.include_router(router=image_router)
app.include_router(router=label_router)
app.include_router(router=export_router)
app.include_router(router=document_router)
app.include_router(router=thread_router)


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
