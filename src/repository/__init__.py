from abc import ABC, abstractmethod
from typing import Annotated

from dependency_injector.wiring import Provide

from src.data.container import DatabaseContainer
from src.data.database import IDatabaseConnection


class IRepository[ID, Entity](ABC):

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> Entity | None:
        """
        Retrieve an entity by its unique identifier.

        This method is expected to be implemented by subclasses to fetch and
        return an entity using its ID. The exact behavior depends on the
        concrete class implementation.

        :param entity_id: The unique identifier of the entity to retrieve.
        :return: The entity associated with the given ID.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """
        Saves the given entity instance asynchronously.

        This method serves as an abstraction for persisting entity data and
        must be implemented by subclasses inheriting this interface. The method
        accepts an entity object and ensures it is stored in the underlying
        data source. The exact process of saving is determined by the concrete
        implementation.

        :param entity: The entity objects to be saved.
        :return: The saved entity object, potentially modified (e.g., assigned an
            identifier or timestamps) upon storing.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_id(self, entity_id: ID) -> Entity | None:
        """
        Deletes an entity with the specified identifier. This method is abstract and
        must be implemented by subclasses.

        :param entity_id: The identifier of the entity to be deleted.
        :return: The deleted entity.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class RepositoryImpl[ID, Entity](IRepository[ID, Entity]):
    connection: Annotated[IDatabaseConnection, Provide[DatabaseContainer.connection]]

    async def get_by_id(self, entity_id: ID) -> Entity | None:
        with self.connection.create_session() as session:
            entity = session.get(Entity, entity_id)
            return entity

    async def save(self, entity: Entity) -> Entity:
        with self.connection.create_session() as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity

    async def delete_by_id(self, entity_id: ID) -> Entity:
        with self.connection.create_session() as session:
            entity = self.get_by_id(entity_id)
            session.delete(entity)
            session.commit()
            return entity
