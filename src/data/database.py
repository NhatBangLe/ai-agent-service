import os

from sqlalchemy import URL
from sqlmodel import SQLModel, create_engine, Session

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "rag_app")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

url = URL.create(
    drivername="postgresql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME
)
engine = create_engine(url)


def get_session():
    """
    Creates and yields a new sqlmodel Session object.

    This function initializes a `Session` using the `engine`.
    The session is closed automatically after use by using the `with` statement.

    Returns:
        Session: A new SQLAlchemy Session object.
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
    SQLModel.metadata.create_all(engine)
