from sqlalchemy import URL
from sqlmodel import SQLModel, create_engine, Session

url = URL.create(
    drivername="postgresql",
    username="postgres",
    password="postgres",
    host="localhost",
    port=5432,
    database="rag_app"
)
engine = create_engine(url)


def get_session():
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
