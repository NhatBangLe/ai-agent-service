import asyncio
import os
import shutil
import zipfile
from pathlib import Path

from .interface.export import IExportingService
from .interface.file import IFileService
from ..data.model import Label, Image
from ..repository.interface.image import IImageRepository
from ..repository.interface.label import ILabelRepository
from ..util.constant import EnvVar
from ..util.error import NotFoundError
from ..util.function import get_datetime_now


class LocalExportingServiceImpl(IExportingService):
    _image_repository: IImageRepository
    _label_repository: ILabelRepository

    def __init__(self,
                 image_repository: IImageRepository,
                 label_repository: ILabelRepository,
                 file_service: IFileService):
        super().__init__()
        self._image_repository = image_repository
        self._label_repository = label_repository
        self._file_service = file_service

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
        current_datetime = get_datetime_now()
        folder_for_exporting = cache_dir.joinpath(label.name)
        file_ext = ".zip"
        exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
        file_info = {
            "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
            "path": str(exported_file.absolute().resolve()),
            "mime_type": "application/zip"
        }

        cache_dir.mkdir(exist_ok=True)

        # Clear old files
        if exported_file.is_file():
            exported_file.unlink(missing_ok=True)
        folder_for_exporting.mkdir()

        for image in images:
            file = await self._file_service.get_file_by_id(image.file_id)
            if file is not None:
                img_file_name = f'{image.id}.{file.name.split('.')[-1]}'
                folder_for_exporting.joinpath(img_file_name).write_bytes(file.data)

        self.zip_folder(folder_for_exporting, exported_file)
        shutil.rmtree(str(folder_for_exporting.absolute().resolve()))

        return IExportingService.ExportedFileMetadata(
            name=file_info["name"],
            path=file_info["path"],
            mime_type=file_info["mime_type"])

    async def export_all_labeled_images(self):
        cache_dir = Path(self.get_cache_dir_path())
        current_datetime = get_datetime_now()
        folder_for_exporting = cache_dir.joinpath("all_labeled_images")
        file_ext = ".zip"
        exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
        file_info = {
            "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
            "path": str(exported_file.absolute().resolve()),
            "mime_type": "application/zip"
        }

        cache_dir.mkdir(exist_ok=True)

        # Clear old files
        if exported_file.is_file():
            exported_file.unlink(missing_ok=True)
        folder_for_exporting.mkdir()

        # Export a new file
        labels_and_images = await self._image_repository.get_all_images_with_labels()
        print(labels_and_images)

        for label, image in labels_and_images:
            label_folder = folder_for_exporting.joinpath(label.name)
            if not label_folder.is_dir():
                label_folder.mkdir()
            file = await self._file_service.get_file_by_id(image.file_id)
            if file is not None:
                img_file_name = f'{image.id}.{file.name.split('.')[-1]}'
                label_folder.joinpath(img_file_name).write_bytes(file.data)

        self.zip_folder(folder_for_exporting, exported_file)
        shutil.rmtree(str(folder_for_exporting.absolute().resolve()))

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
        return os.getenv(EnvVar.CACHE_DIR, "/resource_cache")
