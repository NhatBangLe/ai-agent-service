import datetime
import os
import shutil
from pathlib import Path
from typing import Sequence

from fastapi import APIRouter, status
from sqlalchemy import func
from sqlmodel import Session, select

from .label import get_label
from ..data.model import Image, LabeledImage, Label
from ..dependency import SessionDep, DownloadGeneratorDep
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError
from ..util.function import zip_folder
from ..util.main import SecureDownloadGenerator

DEFAULT_CACHE_DIRECTORY = "/resource"


def _get_cache_dir_path():
    return os.getenv("CACHE_DIR", DEFAULT_CACHE_DIRECTORY)


# noinspection PyTypeChecker
def get_exporting_labeled_images_token(label_id: int, session: Session, generator: SecureDownloadGenerator) -> str:
    label = get_label(label_id, session)
    get_all_images_stmt = (select(Image)
                           .join(LabeledImage, LabeledImage.image_id == Image.id)
                           .where(LabeledImage.label_id == label_id))
    images: Sequence[Image] = session.exec(get_all_images_stmt).all()
    if len(images) == 0:
        raise NotFoundError(f'No assigned images has label with id {label_id}')

    cache_dir = Path(_get_cache_dir_path())
    current_datetime = datetime.datetime.now(DEFAULT_TIMEZONE)
    folder_for_exporting = cache_dir.joinpath(label.name)
    file_ext = ".zip"
    exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
    file_info = {
        "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
        "path": exported_file.absolute().resolve(),
        "mime_type": "application/zip"
    }

    if not cache_dir.is_dir():
        cache_dir.mkdir()
    else:
        # Check whether the old exported file is outdate
        check_outdate_stmt = (select(func.count())
                              .select_from(LabeledImage)
                              .where(LabeledImage.label_id == label_id,
                                     LabeledImage.created_at > current_datetime))
        num_of_new_data = int(session.exec(check_outdate_stmt).one())

        if num_of_new_data == 0 and exported_file.is_file():
            return generator.generate_token(file_info)

    # Clear old folder and files
    if folder_for_exporting.is_dir():
        shutil.rmtree(str(folder_for_exporting.absolute().resolve()))
    if exported_file.is_file():
        exported_file.unlink(missing_ok=True)
    folder_for_exporting.mkdir()

    for image in images:
        image_bytes = Path(image.save_path).read_bytes()
        img_file_name = f'{image.id}.{image.name.split('.')[1]}'
        folder_for_exporting.joinpath(img_file_name).write_bytes(image_bytes)

    zip_folder(folder_for_exporting, exported_file)

    return generator.generate_token(file_info)


# noinspection PyTypeChecker
def get_exporting_all_labeled_images_token(session: Session, generator: SecureDownloadGenerator) -> str:
    cache_dir = Path(_get_cache_dir_path())
    current_datetime = datetime.datetime.now(DEFAULT_TIMEZONE)
    folder_for_exporting = cache_dir.joinpath("all_labeled_images")
    file_ext = ".zip"
    exported_file = folder_for_exporting.with_name(f'{folder_for_exporting.name}{file_ext}')
    file_info = {
        "name": f'{folder_for_exporting.name}_{current_datetime.strftime("%d-%m-%Y_%H-%M-%S")}{file_ext}',
        "path": exported_file.absolute().resolve(),
        "mime_type": "application/zip"
    }

    if not cache_dir.is_dir():
        cache_dir.mkdir()
    else:
        # Check whether the old exported file is outdate
        check_outdate_stmt = (select(func.count())
                              .where(LabeledImage.created_at > current_datetime))
        num_of_new_data = int(session.exec(check_outdate_stmt).one())

        if num_of_new_data == 0 and exported_file.is_file():
            return generator.generate_token(file_info)

    # Clear old folders and files
    if folder_for_exporting.is_dir():
        shutil.rmtree(str(folder_for_exporting.absolute().resolve()))
    if exported_file.is_file():
        exported_file.unlink(missing_ok=True)
    folder_for_exporting.mkdir()

    # Export a new file
    get_all_used_labels_stmt = (select(Label, Image)
                                .join(LabeledImage, LabeledImage.label_id == Label.id)
                                .join(Image, LabeledImage.image_id == Image.id))
    labels_and_images: Sequence[tuple[Label, Image]] = session.exec(get_all_used_labels_stmt).all()

    for label, image in labels_and_images:
        label_folder = folder_for_exporting.joinpath(label.name)
        if not label_folder.is_dir():
            label_folder.mkdir()
        image_bytes = Path(image.save_path).read_bytes()
        img_file_name = f'{image.id}.{image.name.split('.')[1]}'
        label_folder.joinpath(img_file_name).write_bytes(image_bytes)

    zip_folder(folder_for_exporting, exported_file)

    return generator.generate_token(file_info)


router = APIRouter(
    prefix="/api/v1/export",
    tags=["Export"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/all", status_code=status.HTTP_200_OK)
async def get_exporting_all_token(session: SessionDep, generator: DownloadGeneratorDep):
    return get_exporting_all_labeled_images_token(session=session, generator=generator)


@router.get("/{label_id}/label", status_code=status.HTTP_200_OK)
async def export_by_label_id(label_id: int, session: SessionDep, generator: DownloadGeneratorDep):
    return get_exporting_labeled_images_token(label_id=label_id, session=session, generator=generator)
