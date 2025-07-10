import asyncio
import datetime
import os
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any

from dependency_injector.wiring import Provide
from pydantic import BaseModel, Field

from ..data.model import Label, Image
from ..repository.container import RepositoryContainer
from ..repository.image import IImageRepository
from ..repository.label import ILabelRepository
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError


class IExportingService(ABC):
    class ExportedFileMetadata(BaseModel):
        name: str = Field(description="Name of the exported file.")
        path: str = Field(description="Path to the exported file.")
        mime_type: str = Field(description="MIME type of the exported file.")
        kwargs: dict[str, Any] | None = Field(default=None, description="Additional metadata.")

    @abstractmethod
    async def export_labeled_images_by_label_id(self, label_id: int) -> ExportedFileMetadata:
        """
        Exports labeled images associated with the provided label ID as a compressed file.
        This operation retrieves the label and associated images by their label ID.
        The resulting file is returned with its metadata.

        :param label_id: The unique identifier of the label whose associated images
            need to be exported.
        :return: Metadata of the exported file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def export_all_labeled_images(self) -> ExportedFileMetadata:
        """
        Exports all labeled images into a zip archive file.

        This asynchronous method fetches all labeled images through the image repository, organizes
        them into folders corresponding to their labels, and creates a compressed file for
        download. The compressed file is saved in the designated cache directory.

        :returns: Metadata for the exported file.
        :raises NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class LocalExportingServiceImpl(IExportingService):
    image_repository: Annotated[IImageRepository, Provide[RepositoryContainer.image_repository]]
    label_repository: Annotated[ILabelRepository, Provide[RepositoryContainer.label_repository]]

    async def export_labeled_images_by_label_id(self, label_id: int):
        tasks = []
        async with asyncio.TaskGroup() as tg:
            tasks.append(tg.create_task(self.label_repository.get_by_id(label_id)))
            tasks.append(tg.create_task(self.image_repository.get_all_by_label_id(label_id)))
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
        labels_and_images = await self.image_repository.get_all_images_with_labels()

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
