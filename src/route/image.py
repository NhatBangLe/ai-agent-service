import asyncio
from typing import Sequence, Annotated
from uuid import UUID

import PIL.Image
from dependency_injector.wiring import inject
from fastapi import APIRouter, UploadFile, status, Query
from fastapi.responses import FileResponse
from pydantic import Field

from ..data.dto import ImagePublic, ImageCreate
from ..data.model import Image
from ..dependency import PagingQuery, ImageServiceDepend, FileServiceDepend, AgentServiceDepend
from ..process.recognizer.image import ImageRecognizer
from ..service.interface.file import IFileService
from ..service.interface.image import IImageService
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError, InvalidArgumentError
from ..util.function import strict_uuid_parser, get_cache_dir_path


class LabelsWithPagingParams(PagingParams):
    label_ids: Sequence[int] = Field(min_length=1)


async def to_image_public(db_image: Image, file_service: IFileService):
    file = await file_service.get_metadata_by_id(db_image.file_id)
    assigned_label_ids = [rec.label_id for rec in db_image.assigned_labels]
    classified_label_ids = [rec.label_id for rec in db_image.classified_labels]
    return ImagePublic(id=db_image.id, name=file.name,
                       created_at=db_image.created_at,
                       mime_type=file.mime_type,
                       assigned_label_ids=assigned_label_ids,
                       classified_label_ids=classified_label_ids)


async def predict_labels(recognizer: ImageRecognizer,
                         file_bytes: bytes,
                         image_id: UUID,
                         image_service: IImageService) -> None:
    cache_dir = get_cache_dir_path().joinpath("image_to_predict")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir.joinpath(str(image_id))
    cached_file.write_bytes(file_bytes)
    image_file = PIL.Image.open(cached_file)

    result = await recognizer.async_predict(image_file)
    cached_file.unlink()
    await image_service.assign_labels_by_label_names(image_id, list(result["classes"]))


router = APIRouter(
    prefix="/api/v1/images",
    tags=["Images"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/{image_id}/show", status_code=status.HTTP_200_OK)
@inject
async def show(image_id: str, service: ImageServiceDepend, file_service: FileServiceDepend):
    db_image = await service.get_image_by_id(image_id=strict_uuid_parser(image_id))
    file = await file_service.get_metadata_by_id(db_image.file_id)
    if file is None:
        raise NotFoundError(f'Cannot resolve the image with id {image_id}. Because of no found file.')
    return FileResponse(path=file.path, media_type=file.mime_type, filename=file.name)


@router.get("/{image_id}/info", response_model=ImagePublic, status_code=status.HTTP_200_OK)
@inject
async def get_information(image_id: str, service: ImageServiceDepend, file_service: FileServiceDepend):
    image_uuid = strict_uuid_parser(image_id)
    db_image = await service.get_image_by_id(image_uuid)
    return await to_image_public(db_image, file_service)


@router.get("/labels", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_by_label_ids(params: Annotated[LabelsWithPagingParams, Query()],
                           service: ImageServiceDepend, file_service: FileServiceDepend):
    paging = await service.get_images_by_label_ids(params=params, label_ids=params.label_ids)
    return await PagingWrapper.async_convert_content_type(paging, lambda img: to_image_public(img, file_service))


@router.get("/unlabeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_unlabeled(params: PagingQuery, service: ImageServiceDepend, file_service: FileServiceDepend):
    paging = await service.get_unlabeled_images(params=params)
    return await PagingWrapper.async_convert_content_type(paging, lambda img: to_image_public(img, file_service))


@router.get("/labeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_labeled(params: PagingQuery, service: ImageServiceDepend, file_service: FileServiceDepend):
    paging = await service.get_labeled_images(params=params)
    return await PagingWrapper.async_convert_content_type(paging, lambda img: to_image_public(img, file_service))


@router.post("/upload", status_code=status.HTTP_201_CREATED)
@inject
async def upload(file: UploadFile, image_service: ImageServiceDepend, agent_service: AgentServiceDepend) -> str:
    if "image" not in file.content_type:
        raise InvalidArgumentError(f'Unsupported MIME type: {file.content_type}.')
    file_bytes = await file.read()
    image = await image_service.save_image(ImageCreate(name=file.filename,
                                                       mime_type=file.content_type,
                                                       data=file_bytes))

    if agent_service.configurer.image_recognizer is not None:
        asyncio.create_task(
            predict_labels(agent_service.configurer.image_recognizer, file_bytes, image.id, image_service))

    return str(image.id)


@router.post("/{image_id}/assign", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def assign_label(image_id: str, label_ids: list[int], service: ImageServiceDepend) -> None:
    image_uuid = strict_uuid_parser(image_id)
    await service.assign_labels_by_label_ids(image_id=image_uuid, label_ids=label_ids)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(image_id: str, image_service: ImageServiceDepend) -> None:
    image_uuid = strict_uuid_parser(image_id)
    await image_service.delete_image_by_id(image_uuid)
