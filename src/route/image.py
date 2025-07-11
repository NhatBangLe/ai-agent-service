import asyncio
from typing import Sequence, Annotated

from dependency_injector.wiring import inject
from fastapi import APIRouter, UploadFile, status, Query
from fastapi.responses import FileResponse
from pydantic import Field

from ..data.dto import ImagePublic, ImageCreate
from ..dependency import PagingQuery, ImageServiceDepend, FileServiceDepend
from ..service.interface.file import IFileService
from ..util import PagingWrapper, PagingParams
from ..util.error import NotFoundError
from ..util.function import strict_uuid_parser


class LabelsWithPagingParams(PagingParams):
    label_ids: Sequence[int] = Field(min_length=1)


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
    file = await file_service.get_file_by_id(db_image.file_id)
    if file is None:
        raise NotFoundError(f'Cannot resolve the image with id {image_id}. Because of no found file.')
    return FileResponse(path=file.path, media_type=file.mime_type, filename=file.name)


@router.get("/{image_id}/info", response_model=ImagePublic, status_code=status.HTTP_200_OK)
@inject
async def get_information(image_id: str, service: ImageServiceDepend):
    image_uuid = strict_uuid_parser(image_id)
    return await service.get_image_by_id(image_uuid)


@router.get("/labels", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_by_label_ids(params: Annotated[LabelsWithPagingParams, Query()], service: ImageServiceDepend):
    await service.get_images_by_label_ids(params=params, label_ids=params.label_ids)


@router.get("/unlabeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_unlabeled(params: PagingQuery, service: ImageServiceDepend):
    return await service.get_unlabeled_images(params=params)


@router.get("/labeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
@inject
async def get_labeled(params: PagingQuery, service: ImageServiceDepend):
    return await service.get_labeled_images(params=params)


@router.post("/{user_id}/upload", status_code=status.HTTP_201_CREATED)
@inject
async def upload(user_id: str, file: UploadFile,
                 image_service: ImageServiceDepend,
                 file_service: FileServiceDepend) -> str:
    file_bytes = await file.read()
    # Save the uploaded file by using the file service
    save_file = IFileService.SaveFile(name=file.filename, mime_type=file.content_type, data=file_bytes)
    file_id = await file_service.save_file(save_file)
    uploaded_image_id = await image_service.save_image(ImageCreate(user_id=strict_uuid_parser(user_id),
                                                                   file_id=file_id))
    return str(uploaded_image_id)


@router.post("/{image_id}/assign", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def assign_label(image_id: str, label_ids: list[int], service: ImageServiceDepend) -> None:
    image_uuid = strict_uuid_parser(image_id)
    await service.assign_labels(image_id=image_uuid, label_ids=label_ids)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(image_id: str, image_service: ImageServiceDepend, file_service: FileServiceDepend) -> None:
    image_uuid = strict_uuid_parser(image_id)
    image = await image_service.get_image_by_id(image_uuid)
    async with asyncio.TaskGroup() as tg:
        tg.create_task(file_service.delete_file_by_id(image.file_id))
        tg.create_task(image_service.delete_image(image))
