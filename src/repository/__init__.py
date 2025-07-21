from abc import ABC
from typing import Iterable

from sqlmodel import Session

from ..data.database import IDatabaseConnection
from ..repository.interface import IRepository


class RepositoryImpl[ID, Entity](IRepository[ID, Entity], ABC):
    _connection: IDatabaseConnection

    def __init__(self, connection: IDatabaseConnection):
        super().__init__()
        self._connection = connection

    async def get_session(self) -> Session:
        return self._connection.create_session()

    async def save(self, entity: Entity) -> Entity:
        with self._connection.create_session() as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity

    async def save_all(self, entities: Iterable[Entity]) -> None:
        with self._connection.create_session() as session:
            session.add_all(entities)
            session.commit()

    async def delete(self, entity: Entity) -> None:
        with self._connection.create_session() as session:
            session.delete(entity)
            session.commit()

    async def delete_by_id(self, entity_id: ID) -> Entity | None:
        with self._connection.create_session() as session:
            entity = await self.get_by_id(entity_id)
            if entity is None:
                return None
            session.delete(entity)
            session.commit()
            return entity
