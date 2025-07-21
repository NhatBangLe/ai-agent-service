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
from src.data.database import DatabaseConnection
from src.dependency import DownloadGeneratorDepend
from src.repository.document import DocumentRepositoryImpl
from src.repository.file import FileRepositoryImpl
from src.repository.image import ImageRepositoryImpl
from src.repository.label import LabelRepositoryImpl
from src.repository.thread import ThreadRepositoryImpl
from src.route.agent import router as agent_router
from src.route.document import router as document_router
from src.route.export import router as export_router
from src.route.image import router as image_router
from src.route.label import router as label_router
from src.route.thread import router as thread_router
from src.service.document import DocumentServiceImpl
from src.service.export import LocalExportingServiceImpl
from src.service.file import LocalFileService
from src.service.image import ImageServiceImpl
from src.service.label import LabelServiceImpl
from src.service.thread import ThreadServiceImpl
from src.container import ApplicationContainer
from src.util.constant import EnvVar
from src.util.error import NotFoundError, InvalidArgumentError


# Set up logging.
def setup_logging():
    level = os.getenv(EnvVar.LOG_LEVEL.value, "INFO")
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


async def init_application_container():
    # Database
    db_connection = providers.Resource(DatabaseConnection,
                                       host=os.getenv(EnvVar.DB_HOST.value, "localhost"),
                                       port=os.getenv(EnvVar.DB_PORT.value, "5432"),
                                       user=os.getenv(EnvVar.DB_USER.value, "postgres"),
                                       password=os.getenv(EnvVar.DB_PASSWORD.value, "postgres"),
                                       database=os.getenv(EnvVar.DB_NAME.value, "rag_app"))

    # Repositories
    image_repository = providers.Singleton(ImageRepositoryImpl, connection=db_connection)
    label_repository = providers.Singleton(LabelRepositoryImpl, connection=db_connection)
    document_repository = providers.Singleton(DocumentRepositoryImpl, connection=db_connection)
    file_repository = providers.Singleton(FileRepositoryImpl, connection=db_connection)
    thread_repository = providers.Singleton(ThreadRepositoryImpl, connection=db_connection)

    # Services
    file_service = providers.Singleton(LocalFileService, file_repository=file_repository)
    image_service = providers.Singleton(ImageServiceImpl, image_repository=image_repository,
                                        label_repository=label_repository)
    document_service = providers.Singleton(DocumentServiceImpl, document_repository=document_repository)
    label_service = providers.Singleton(LabelServiceImpl, label_repository=label_repository)
    thread_service = providers.Singleton(ThreadServiceImpl, thread_repository=thread_repository)
    exporting_service = providers.Singleton(LocalExportingServiceImpl, image_repository=image_repository,
                                            label_repository=label_repository)

    container = ApplicationContainer(db_connection=db_connection,
                                     image_repository=image_repository,
                                     label_repository=label_repository,
                                     document_repository=document_repository,
                                     file_repository=file_repository,
                                     thread_repository=thread_repository,
                                     file_service=file_service,
                                     image_service=image_service,
                                     document_service=document_service,
                                     label_service=label_service,
                                     thread_service=thread_service,
                                     exporting_service=exporting_service)
    repository_modules = [".repository.document", ".repository.file", ".repository.image", ".repository.label",
                          ".repository.thread"]
    service_modules = [".service.document", ".service.export", ".service.file", ".service.image", ".service.label",
                       ".service.thread"]
    route_modules = [".route.document", ".route.export", ".route.image", ".route.label", ".route.thread"]
    agent_modules = [".config.configurer.agent", ".config.configurer.bm25", ".config.configurer.recognizer.image"]
    container.wire(modules=[*repository_modules, *service_modules, *route_modules, *agent_modules])
    return container


async def shutdown_application_container(container: ApplicationContainer):
    awaitable = container.shutdown_resources()
    if awaitable:
        await awaitable


@asynccontextmanager
async def lifespan(api: FastAPI):
    container = await init_application_container()
    api.container = container
    container.db_connection().create_db_and_tables()

    # Initialize the agent.
    await agent.configure()
    agent.build_graph()

    yield

    await agent.shutdown()
    await shutdown_application_container(container)


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
async def download(token: str, generator: DownloadGeneratorDepend):
    file = generator.verify_token(token)
    if file is None:
        raise InvalidArgumentError("Invalid token.")
    return FileResponse(path=file["path"], media_type=file["mime_type"], filename=file["name"])


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
