import asyncio
import datetime
import os
import shutil
import zipfile
from pathlib import Path

from .interface.export import IExportingService
from ..data.model import Label, Image
from ..repository.interface.image import IImageRepository
from ..repository.interface.label import ILabelRepository
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError


class LocalExportingServiceImpl(IExportingService):
    _image_repository: IImageRepository
    _label_repository: ILabelRepository

    def __init__(self, image_repository: IImageRepository, label_repository: ILabelRepository):
        super().__init__()
        self._image_repository = image_repository
        self._label_repository = label_repository

    async def export_labeled_images_by_label_id(self, label_id: int):
        tasks = []
        async with asyncio.TaskGroup() as tg:
            tasks.append(tg.create_task(self._label_repository.get_by_id(label_id)))
            tasks.append(tg.create_task(self._image_repository.get_all_by_label_id(label_id)))
        label: Label | None = tasks[0].result()
        images: list[Image] = tasks[1].result()
        if len(images) == 0:
            raise NotFoundError(f'No assigned images has label with id {label_id}')

        cache_dir = Path(self.get_cache_dir_path())
        current_datetime = datetime.datetime.now(DEFAULT_TIMEZONE)
        folder_for_exporting = cache_dir.joinpath(label.name)
        file_ext = ".zip"
        exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
        file_info = {
            "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
            "path": exported_file.absolute().resolve(),
            "mime_type": "application/zip"
        }

        cache_dir.mkdir(exist_ok=True)

        # Clear old folder and files
        if folder_for_exporting.is_dir():
            shutil.rmtree(str(folder_for_exporting.absolute().resolve()))
        if exported_file.is_file():
            exported_file.unlink(missing_ok=True)
        folder_for_exporting.mkdir()

        for image in images:
            image_bytes = Path(image.save_path).read_bytes()
            img_file_name = f'{image.id}.{image.name.split('.')[-1]}'
            folder_for_exporting.joinpath(img_file_name).write_bytes(image_bytes)

        self.zip_folder(folder_for_exporting, exported_file)
        return IExportingService.ExportedFileMetadata(
            name=file_info["name"],
            path=file_info["path"],
            mime_type=file_info["mime_type"])

    async def export_all_labeled_images(self):
        cache_dir = Path(self.get_cache_dir_path())
        current_datetime = datetime.datetime.now(DEFAULT_TIMEZONE)
        folder_for_exporting = cache_dir.joinpath("all_labeled_images")
        file_ext = ".zip"
        exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
        file_info = {
            "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
            "path": exported_file.absolute().resolve(),
            "mime_type": "application/zip"
        }

        cache_dir.mkdir(exist_ok=True)

        # Clear old folders and files
        if folder_for_exporting.is_dir():
            shutil.rmtree(str(folder_for_exporting.absolute().resolve()))
        if exported_file.is_file():
            exported_file.unlink(missing_ok=True)
        folder_for_exporting.mkdir()

        # Export a new file
        labels_and_images = await self._image_repository.get_all_images_with_labels()

        for label, image in labels_and_images:
            label_folder = folder_for_exporting.joinpath(label.name)
            if not label_folder.is_dir():
                label_folder.mkdir()
            image_bytes = Path(image.save_path).read_bytes()
            img_file_name = f'{image.id}.{image.name.split('.')[-1]}'
            label_folder.joinpath(img_file_name).write_bytes(image_bytes)

        self.zip_folder(folder_for_exporting, exported_file)
        return IExportingService.ExportedFileMetadata(
            name=file_info["name"],
            path=file_info["path"],
            mime_type=file_info["mime_type"])

    @staticmethod
    def zip_folder(folder_path: str | os.PathLike[str], output_path: str | os.PathLike[str]):
        """
        Zip a folder
        :param folder_path: Path to a folder which needs to archive
        :param output_path: Path to the zip output file
        """
        folder = Path(folder_path)
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in folder.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(folder))

    @staticmethod
    def get_cache_dir_path():
        return os.getenv("CACHE_DIR", "/resource_cache")
