import datetime
import logging
import os
from os import PathLike
from pathlib import Path

from sqlalchemy import URL
from sqlmodel import SQLModel, create_engine, Session

from .base_model import DocumentSource
from ..config.model.data import ExternalDocumentConfiguration
from ..data.model import Label, Document, DocumentChunk
from ..process.recognizer import RecognizerOutput
from ..util.constant import DEFAULT_TIMEZONE

DATABASE_HOST_ENV = "DB_HOST"
DATABASE_PORT_ENV = "DB_PORT"
DATABASE_NAME_ENV = "DB_NAME"
DATABASE_USER_ENV = "DB_USER"
DATABASE_PASSWORD_ENV = "DB_PASSWORD"

logger = logging.getLogger("database_logger")

url = URL.create(
    drivername="postgresql+psycopg",
    host=os.getenv(DATABASE_HOST_ENV, "localhost"),
    port=int(os.getenv(DATABASE_PORT_ENV, "5432")),
    username=os.getenv(DATABASE_USER_ENV, "postgres"),
    password=os.getenv(DATABASE_PASSWORD_ENV, "postgres"),
    database=os.getenv(DATABASE_NAME_ENV, "rag_app")
)
engine = create_engine(url)


def get_session():
    """
    Creates and yields a new sqlmodel Session object.

    This function initializes a `Session` using the `engine`.
    The session is closed automatically after use by using the `with` statement.
    """
    with Session(engine) as session:
        yield session


def create_session():
    """
    Creates and returns a new sqlmodel Session object.

    This function initializes a `Session` using the `engine`.

    **Important: ** It is crucial to close the session after it has been used
    to release database connections and resources. This can be done using a
    `try...finally` block or, preferably, a `with` statement for automatic
    resource management.

    Returns:
        Session: A new SQLAlchemy Session object.
    """
    return Session(engine)


def create_db_and_tables():
    """
    Creates all database tables defined in the SQLModel metadata.

    This method is typically called during application initialization to ensure
    that the necessary database schema exists before any data operations are performed.
    It uses the metadata associated with SQLModel to create tables based on
    defined models.

    Raises:
        SQLAlchemyError: If there is an issue connecting to the database or
                         creating the tables (e.g., permissions issues, invalid connection string).
    """
    SQLModel.metadata.create_all(engine)


def insert_predefined_output_classes(config_file_path: str | PathLike[str]):
    """
    Initializes the application's database with predefined labels from a configuration file.
    This process is designed to run only once to prevent duplicate data insertion.

    Args:
        config_file_path: A absolute path to the configuration file containing the predefined labels.

    Raises:
        FileNotFoundError: If the configuration file specified by `config.output_config_path`
                           does not exist.
        IOError: If there's an issue reading from or writing to the configuration file.
        DatabaseError: If there's an issue interacting with the database (e.g., adding labels, committing).
        ValidationError: If the content of the configuration file does not conform to the
                         `RecognizerOutput` model's expected structure.
    """
    logger.debug(f"Reading and validating predefined output classes from config file: {config_file_path}")
    file_path = Path(config_file_path)
    json_bytes = file_path.read_bytes()
    output = RecognizerOutput.model_validate_json(json_bytes)
    if output.is_configured:
        logger.debug("Predefined output classes are already configured. Skipping...")
        return

    logger.debug(f'Saving predefined output classes to database...')
    with create_session() as session:
        for output_class in output.classes:
            label = Label(name=output_class.name, description=output_class.description)
            session.add(label)
        session.commit()

    classes = "\n".join(
        f'name: {output_class.name}, desc: {output_class.description}'
        for output_class in output.classes)
    logger.debug(f"Labels saved to database: {classes}")

    logger.debug(f'Updating config file with configured status: is_configured = True...')
    with open(config_file_path, 'w'):  # Clear old content
        pass
    output.is_configured = True  # Mark as configured
    file_path.write_text(output.model_dump_json(indent=2))


def insert_external_data(ext_data_file_path: str | PathLike[str]):
    json_bytes = Path(ext_data_file_path).read_bytes()
    config = ExternalDocumentConfiguration.model_validate_json(json_bytes)

    with create_session() as session:
        for store in config.vector_stores:
            for d in store.documents:
                db_chunks = [DocumentChunk(id=str(chunk_id)) for chunk_id in d.chunk_ids]
                db_doc = Document(
                    created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
                    name=d.name,
                    source=DocumentSource.EXTERNAL,
                    embed_to_vs=store.name,
                    chunks=db_chunks)
                session.add(db_doc)
        session.commit()
