import datetime
from abc import ABC, abstractmethod
from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide
from fastapi import UploadFile

from .container import ServiceContainer
from .file import IFileService
from ..data.model import Image
from ..repository.container import RepositoryContainer
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError


class IImageService(ABC):

    @abstractmethod
    async def get_image_by_id(self, image_id: UUID) -> Image:
        raise NotImplementedError

    @abstractmethod
    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        raise NotImplementedError

    @abstractmethod
    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        raise NotImplementedError


class ImageServiceImpl(IImageService):
    image_repository: Annotated[IImageRepository, Provide[RepositoryContainer.image_repository]]
    label_repository: Annotated[ILabelRepository, Provide[RepositoryContainer.label_repository]]
    file_service: Annotated[IFileService, Provide[ServiceContainer.file_service]]

    async def get_image_by_id(self, image_id: UUID) -> Image:
        db_image = await self.image_repository.get_by_id(entity_id=image_id)
        if db_image is None:
            raise NotFoundError(f'No image with id {image_id} found.')
        return db_image

    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        file_bytes = await file.read()
        save_file = IFileService.SaveFile(name=file.filename, mime_type=file.content_type, data=file_bytes)
        metadata = await self.file_service.save_file(save_file)
        db_image = await self.image_repository.save(Image(
            created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
            name=file.filename,
            mime_type=file.content_type,
            save_path=metadata.path,
            user_id=user_id
        ))
        return db_image.id

    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        db_image = await self.image_repository.get_by_id(entity_id=image_id)
        await self.label_repository.assign_labels(db_image, label_ids)
