from dependency_injector import containers, providers

from .database import DatabaseConnection, IDatabaseConnection


class DatabaseContainer(containers.DeclarativeContainer):
    config = providers.Configuration("database_config")
    connection = providers.Dependency(instance_of=IDatabaseConnection,
                                      default=providers.Resource(DatabaseConnection,
                                                                 host=config.host,
                                                                 port=config.port,
                                                                 user=config.user,
                                                                 password=config.password,
                                                                 database=config.database))
