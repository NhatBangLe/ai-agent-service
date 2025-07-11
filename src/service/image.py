from typing import Annotated
from uuid import UUID

from dependency_injector.wiring import Provide
from fastapi import UploadFile

from .interface.file import IFileService
from .interface.image import IImageService
from ..container import ApplicationContainer
from ..data.model import Image
from ..repository.interface.image import IImageRepository
from ..repository.interface.label import ILabelRepository
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError


class ImageServiceImpl(IImageService):
    image_repository: Annotated[IImageRepository, Provide[ApplicationContainer.repository_container.image_repository]]
    label_repository: Annotated[ILabelRepository, Provide[ApplicationContainer.repository_container.label_repository]]
    file_service: Annotated[IFileService, Provide[ApplicationContainer.service_container.file_service]]

    async def get_image_by_id(self, image_id: UUID) -> Image:
        db_image = await self.image_repository.get_by_id(entity_id=image_id)
        if db_image is None:
            raise NotFoundError(f'No image with id {image_id} found.')
        return db_image

    async def get_images_by_label_ids(self, params: PagingParams, label_ids: list[int]) -> PagingWrapper[Image]:
        return await self.image_repository.get_by_label_ids(params, label_ids)

    async def get_unlabeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        return await self.image_repository.get_unlabeled(params)

    async def get_labeled_images(self, params: PagingParams) -> PagingWrapper[Image]:
        return await self.image_repository.get_labeled(params)

    async def save_image(self, user_id: UUID, file: UploadFile) -> UUID:
        file_bytes = await file.read()
        save_file = IFileService.SaveFile(name=file.filename, mime_type=file.content_type, data=file_bytes)
        saved_file_id = await self.file_service.save_file(save_file)
        db_image = await self.image_repository.save(Image(user_id=user_id, file_id=saved_file_id))
        return db_image.id

    async def delete_image_by_id(self, image_id: UUID) -> Image:
        deleted_image = await self.image_repository.delete_by_id(image_id)
        if deleted_image is None:
            raise NotFoundError(f'Image with id {image_id} not found.')
        await self.file_service.delete_file_by_id(deleted_image.file_id)
        return deleted_image

    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        db_image = await self.get_image_by_id(image_id)
        await self.label_repository.assign_labels(db_image, label_ids)
