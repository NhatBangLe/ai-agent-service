import datetime
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path

from sqlalchemy import URL, Engine
from sqlmodel import SQLModel, create_engine, Session

from .base_model import DocumentSource, LabelSource
from ..config.model.data import ExternalDocumentConfiguration
from ..data.model import Label, Document, DocumentChunk
from ..process.recognizer import RecognizerOutput
from ..util.constant import DEFAULT_TIMEZONE


class IDatabaseConnection(ABC):

    @abstractmethod
    def create_session(self) -> Session:
        """
        Creates and returns a new sqlmodel Session object.

        This function initializes a `Session` using the `engine`.

        **Important: ** It is crucial to close the session after it has been used
        to release database connections and resources. This can be done using a
        `try...finally` block or, preferably, a `with` statement for automatic
        resource management.

        Returns:
            Session: A new SQLAlchemy Session object.
        Raises:
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def create_db_and_tables(self) -> None:
        """
        Creates all database tables defined in the SQLModel metadata.

        This method is typically called during application initialization to ensure
        that the necessary database schema exists before any data operations are performed.
        It uses the metadata associated with SQLModel to create tables based on
        defined models.

        Raises:
            SQLAlchemyError: If there is an issue connecting to the database or
                             creating the tables (e.g., permissions issues, invalid connection string).
            NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError


class DatabaseConnection(IDatabaseConnection):
    _engine: Engine
    url: URL

    def __init__(self, host, port, database, user, password):
        self.url = URL.create(
            drivername="postgresql+psycopg",
            host=host,
            port=port,
            username=user,
            password=password,
            database=database
        )

    def __enter__(self):
        print(f"Connecting to database url: {self.url.render_as_string(hide_password=True)}")
        self._engine = create_engine(self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        self._engine.dispose()

    def create_session(self):
        return Session(self._engine)

    def create_db_and_tables(self):
        SQLModel.metadata.create_all(self._engine)

# def insert_predefined_output_classes(config_file_path: str | PathLike[str]):
#     """
#     Initializes the application's database with predefined labels from a configuration file.
#     This process is designed to run only once to prevent duplicate data insertion.
#
#     Args:
#         config_file_path: A absolute path to the configuration file containing the predefined labels.
#
#     Raises:
#         FileNotFoundError: If the configuration file specified by `config.output_config_path`
#                            does not exist.
#         IOError: If there's an issue reading from or writing to the configuration file.
#         DatabaseError: If there's an issue interacting with the database (e.g., adding labels, committing).
#         ValidationError: If the content of the configuration file does not conform to the
#                          `RecognizerOutput` model's expected structure.
#     """
#     logger.debug(f"Reading and validating predefined output classes from config file: {config_file_path}")
#     file_path = Path(config_file_path)
#     json_bytes = file_path.read_bytes()
#     output = RecognizerOutput.model_validate_json(json_bytes)
#     if output.is_configured:
#         logger.debug("Predefined output classes are already configured. Skipping...")
#         return
#
#     logger.debug(f'Saving predefined output classes to database...')
#     with create_session() as session:
#         for output_class in output.classes:
#             label = Label(name=output_class.name,
#                           description=output_class.description,
#                           source=LabelSource.PREDEFINED)
#             session.add(label)
#         session.commit()
#
#     classes = "\n".join(
#         f'name: {output_class.name}, desc: {output_class.description}'
#         for output_class in output.classes)
#     logger.debug(f"Labels are saved to database: {classes}")
#
#     logger.debug(f'Updating config file with configured status: is_configured = True...')
#     with open(config_file_path, 'w'):  # Clear old content
#         pass
#     output.is_configured = True  # Mark as configured
#     file_path.write_text(output.model_dump_json(indent=2))
#
#
# def insert_external_data(store_name: str, ext_data_file_path: str | PathLike[str]):
#     file_path = Path(ext_data_file_path)
#     json_bytes = file_path.read_bytes()
#     config = ExternalDocumentConfiguration.model_validate_json(json_bytes)
#     if config.is_configured:
#         logger.debug("External data are already configured. Skipping...")
#         return
#
#     with create_session() as session:
#         for d in config.documents:
#             db_chunks = [DocumentChunk(id=str(chunk_id)) for chunk_id in d.chunk_ids]
#             db_doc = Document(
#                 created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
#                 name=d.name,
#                 source=DocumentSource.EXTERNAL,
#                 embed_to_vs=store_name,
#                 chunks=db_chunks)
#             session.add(db_doc)
#         session.commit()
#
#     logger.debug(f'Updating external data config file with configured status: is_configured = True...')
#     with open(ext_data_file_path, 'w'):  # Clear old content
#         pass
#     config.is_configured = True  # Mark as configured
#     file_path.write_text(config.model_dump_json(indent=2))
