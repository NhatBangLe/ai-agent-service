from uuid import UUID

from .interface.image import IImageService
from ..data.dto import ImageCreate
from ..data.model import Image
from ..repository.interface.image import IImageRepository
from ..repository.interface.label import ILabelRepository
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError


class ImageServiceImpl(IImageService):
    image_repository: IImageRepository
    label_repository: ILabelRepository

    def __init__(self, image_repository: IImageRepository, label_repository: ILabelRepository):
        super().__init__()
        self.image_repository = image_repository
        self.label_repository = label_repository

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

    async def save_image(self, data: ImageCreate) -> UUID:
        db_image = await self.image_repository.save(Image(user_id=data.user_id,
                                                          file_id=data.file_id))
        return db_image.id

    async def delete_image_by_id(self, image_id: UUID) -> Image:
        image = await self.get_image_by_id(image_id)
        await self.delete_image(image)
        return image

    async def delete_image(self, image: Image) -> None:
        await self.image_repository.delete(image)

    async def assign_labels(self, image_id: UUID, label_ids: list[int]) -> None:
        db_image = await self.get_image_by_id(image_id)
        await self.label_repository.assign_labels(db_image, label_ids)
