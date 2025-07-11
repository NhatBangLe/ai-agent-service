from abc import ABC, abstractmethod

from sqlalchemy import URL, Engine
from sqlmodel import SQLModel, create_engine, Session


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

    @abstractmethod
    def get_url(self) -> URL:
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

    def get_url(self):
        return self.url
