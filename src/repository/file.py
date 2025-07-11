from . import RepositoryImpl
from .interface.file import IFileRepository


class FileRepositoryImpl(IFileRepository, RepositoryImpl):
    pass
