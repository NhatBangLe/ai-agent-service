import asyncio

from .interface.file import IFileService
from .interface.image import IImageService
from ..data.model import Image, ClassifiedImage
from ..repository.interface.image import IImageRepository
from ..repository.interface.label import ILabelRepository
from ..util.error import NotFoundError


class ImageServiceImpl(IImageService):
    _image_repository: IImageRepository
    _label_repository: ILabelRepository
    _file_service: IFileService

    def __init__(self,
                 image_repository: IImageRepository,
                 label_repository: ILabelRepository,
                 file_service: IFileService):
        super().__init__()
        self._image_repository = image_repository
        self._label_repository = label_repository
        self._file_service = file_service

    async def get_image_by_id(self, image_id):
        db_image = await self._image_repository.get_by_id(entity_id=image_id)
        if db_image is None:
            raise NotFoundError(f'No image with id {image_id} found.')
        return db_image

    async def get_images_by_label_ids(self, params, label_ids):
        return await self._image_repository.get_by_label_ids(params, label_ids)

    async def get_unlabeled_images(self, params):
        return await self._image_repository.get_unlabeled(params)

    async def get_labeled_images(self, params):
        return await self._image_repository.get_labeled(params)

    async def save_image(self, data):
        file = IFileService.SaveFile(name=data.name, data=data.data, mime_type=data.mime_type)
        file_metadata = await self._file_service.save_file(file)
        return await self._image_repository.save(Image(file_id=file_metadata.id))

    async def delete_image_by_id(self, image_id):
        image = await self.get_image_by_id(image_id)
        await self.delete_image(image)
        return image

    async def delete_image(self, image):
        file_id = image.file_id
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._file_service.delete_file_by_id(file_id))
            tg.create_task(self._image_repository.delete(image))

    async def assign_labels_by_label_ids(self, image_id, label_ids):
        db_image = await self.get_image_by_id(image_id)
        await self._label_repository.assign_labels(db_image, label_ids)

    async def assign_labels_by_label_names(self, image_id, label_names):
        async with asyncio.TaskGroup() as tg:
            get_image_task = tg.create_task(self.get_image_by_id(image_id))
            get_labels_task = tg.create_task(self._label_repository.get_in_names(label_names))
        db_image = get_image_task.result()
        labels = get_labels_task.result()

        classified_labels = [ClassifiedImage(label=label, image=db_image) for label in labels]
        db_image.classified_labels = classified_labels
        await self._image_repository.save(db_image)
