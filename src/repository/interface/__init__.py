from abc import ABC, abstractmethod
from typing import Iterable


class IRepository[ID, Entity](ABC):

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> Entity | None:
        """
        Retrieve an entity by its unique identifier.

        This method interacts with the database to fetch an entity corresponding
        to the given unique identifier. If no matching entity is found, it returns
        None.

        :param entity_id: The unique identifier of the entity to retrieve.
        :return: The entity instance if found, otherwise None.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """
        Asynchronously saves an entity to the database.

        This method saves the provided entity to the database by creating a new
        session, adding the entity, committing the transaction, and refreshing
        the entity to ensure it has the most up-to-date state. The saved entity
        is then returned.

        :param entity: The entity to be saved to the database.
        :return: The saved entity with its updated state.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def save_all(self, entities: Iterable[Entity]) -> None:
        """
        Saves all provided entities to the database asynchronously.

        This method accepts an `Iterable` of entities and attempts to save them
        within a database session. It ensures that all entities are added to
        the session and the changes are committed.

        :param entities: An iterable collection of Entity objects to be saved.
        :return: This function does not return any value.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete(self, entity: Entity) -> None:
        """
        Deletes a given entity from the database using the active session.
        This method uses a session created from the current database connection
        and commits the transaction after deletion.

        :param entity: The entity to be removed from the database. Must be
            an instance of the `Entity` model.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_id(self, entity_id: ID) -> Entity | None:
        """
        Deletes an entity with the given ID from the database and returns the deleted entity.

        This method retrieves an entity by its ID, deletes it from the database using an
        active session, commits the transaction, and then returns the deleted entity.

        :param entity_id: The unique identifier of the entity to be deleted.
        :return: The entity that was deleted.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError
