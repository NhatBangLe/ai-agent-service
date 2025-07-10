import asyncio
import logging
import os
import platform
from contextlib import asynccontextmanager

from dependency_injector import providers
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from src.agent.agent import Agent
from src.config.configurer.agent import AgentConfigurer
from src.container import ApplicationContainer
from src.data.container import DatabaseContainer
from src.dependency import DownloadGeneratorDep
from src.route.agent import router as agent_router
from src.route.document import router as document_router
from src.route.export import router as export_router
from src.route.image import router as image_router
from src.route.label import router as label_router
from src.route.thread import router as thread_router
from src.service.file import LocalFileService
from src.service.image import ImageServiceImpl
from src.util.error import NotFoundError, InvalidArgumentError


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
    return matches[level]


def setup_event_loop():
    if 'Windows' in platform.system():
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)


# Initialize
load_dotenv()
setup_event_loop()
logging_level = setup_logging()
configurer = AgentConfigurer()
agent = Agent(configurer=configurer)


def get_agent():
    agent.check_graph_available()
    return agent


def configure_database(container: DatabaseContainer):
    container.config.host.from_env("DB_HOST", "localhost")
    container.config.port.from_env("DB_PORT", "5432")
    container.config.database.from_env("DB_NAME", "rag_app")
    container.config.username.from_env("DB_USER", "postgres")
    container.config.password.from_env("DB_PASSWORD", "postgres")


@asynccontextmanager
async def lifespan(api: FastAPI):
    container = ApplicationContainer(
        file_service=providers.Singleton(LocalFileService),
        image_service=providers.Singleton(ImageServiceImpl),
    )
    api.container = container

    db_container = container.database_container()
    configure_database(db_container)
    await db_container.init_resources()
    db_container.connection().create_db_and_tables()

    container.wire(modules=[".repository.image", ".repository.label", ".service.image"])

    # Initialize the agent.
    await agent.configure()
    agent.build_graph()

    # agent_config = agent.configurer.config
    # # Insert predefined output classes to the database.
    # if agent_config.image_recognizer is not None:
    #     recognizer_output_config_path = agent_config.image_recognizer.output_config_path
    #     config_file_path = os.path.join(get_config_folder_path(), recognizer_output_config_path)
    #     insert_predefined_output_classes(str(config_file_path))

    # # Insert external data from vector stores
    # retriever_configs = agent_config.retrievers
    # if retriever_configs is not None:
    #     vs_configs: list[VectorStoreConfiguration] = list(
    #         filter(lambda config: isinstance(config, VectorStoreConfiguration), retriever_configs))
    #     for c in vs_configs:
    #         path = c.external_data_config_path
    #         if path is not None:
    #             config_file_path = os.path.join(get_config_folder_path(), path)
    #             insert_external_data(store_name=c.name, ext_data_file_path=str(config_file_path))

    yield

    await agent.shutdown()
    await db_container.shutdown_resources()


app = FastAPI(lifespan=lifespan, debug=logging_level == logging.DEBUG)
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
