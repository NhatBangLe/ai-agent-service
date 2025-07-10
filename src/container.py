from dependency_injector import containers, providers

from .data.container import DatabaseContainer
from .repository.container import RepositoryContainer
from .service.container import ServiceContainer


class ApplicationContainer(containers.DeclarativeContainer):
    database_container = providers.Container(DatabaseContainer)
    repository_container = providers.Container(RepositoryContainer, db_connection=database_container.connection)
    service_container = providers.Container(ServiceContainer,
                                            image_repository=repository_container.image_repository,
                                            label_repository=repository_container.label_repository,
                                            document_repository=repository_container.document_repository)
