import datetime
import os.path
import typing
from pathlib import Path
from typing import Sequence, Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, status, Query
from fastapi.responses import FileResponse
from pydantic import Field
from sqlalchemy import func
from sqlmodel import Session, select

from .label import get_label
from ..data.dto import ImagePublic, PagingWrapper
from ..data.model import Image, User, LabeledImage
from ..dependency import SessionDep, PagingQuery, PagingParams
from ..util.constant import DEFAULT_TIMEZONE
from ..util.error import NotFoundError
from ..util.function import strict_uuid_parser, get_paging
from ..util import FileInformation, SecureDownloadGenerator

DEFAULT_SAVE_DIRECTORY = "/resource"


def _get_save_image_directory():
    return os.getenv("SAVE_IMAGE_DIRECTORY", DEFAULT_SAVE_DIRECTORY)


def get_image(image_id: UUID, session: Session) -> Image:
    db_image = session.get(Image, image_id)
    if db_image is None:
        raise NotFoundError(f'Image with id {image_id} not found.')
    return typing.cast(Image, db_image)


def get_image_download_token(image_id: UUID, session: Session, generator: SecureDownloadGenerator) -> str:
    db_image = get_image(image_id, session)
    data: FileInformation = {
        "name": db_image.name,
        "mime_type": db_image.mime_type,
        "path": db_image.save_path,
    }
    return generator.generate_token(data=data)


# noinspection PyTypeChecker,PyComparisonWithNone
def get_images_by_label_id(label_id: int, params: PagingParams, session: Session) -> list[Image]:
    statement = (select(Image)
                 .distinct()
                 .join(LabeledImage, LabeledImage.image_id == Image.id)
                 .where(LabeledImage.label_id == label_id)
                 .offset((params.offset * params.limit))
                 .limit(params.limit)
                 .order_by(Image.created_at))
    results = session.exec(statement)
    return list(results.all())


class LabelsWithPagingParams(PagingParams):
    label_ids: Sequence[int] = Field(min_length=1)


# noinspection PyTypeChecker,PyUnresolvedReferences
def get_images_by_label_ids(params: LabelsWithPagingParams, session: Session) -> PagingWrapper[Image]:
    subquery = (select(LabeledImage.image_id)
                .where(LabeledImage.label_id.in_(params.label_ids))
                .group_by(LabeledImage.image_id)
                .having(func.count(LabeledImage.label_id) == len(params.label_ids))
                .subquery())

    count_statement = (select(func.count())
                       .select_from(Image)
                       .where(Image.id.in_(select(subquery.c.image_id))))

    statement = (select(Image)
                 .where(Image.id.in_(select(subquery.c.image_id)))
                 .offset(params.offset * params.limit)
                 .limit(params.limit)
                 .order_by(Image.created_at))

    return get_paging(
        params=params,
        count_statement=count_statement,
        execute_statement=statement,
        session=session
    )


# noinspection PyTypeChecker,PyComparisonWithNone
def get_unlabeled_images(params: PagingParams, session: Session) -> PagingWrapper[Image]:
    count_statement = (select(func.count())
                       .outerjoin_from(Image, LabeledImage, LabeledImage.image_id == Image.id)
                       .where(LabeledImage.label_id == None))

    statement = (select(Image)
                 .join(LabeledImage, LabeledImage.image_id == Image.id, isouter=True)
                 .where(LabeledImage.label_id == None)
                 .offset(params.offset)
                 .limit(params.limit)
                 .order_by(Image.created_at))
    return get_paging(
        params=params,
        count_statement=count_statement,
        execute_statement=statement,
        session=session
    )


# noinspection PyTypeChecker
def get_labeled_images(params: PagingParams, session: Session) -> PagingWrapper[Image]:
    count_statement = (select(func.count(func.distinct(Image.id)))
                       .select_from(Image)
                       .join(LabeledImage, LabeledImage.image_id == Image.id))
    statement = (select(Image)
                 .distinct()
                 .join(LabeledImage, LabeledImage.image_id == Image.id)
                 .offset((params.offset * params.limit))
                 .limit(params.limit)
                 .order_by(Image.created_at))
    return get_paging(
        params=params,
        count_statement=count_statement,
        execute_statement=statement,
        session=session
    )


async def save_image(user_id: UUID, file: UploadFile, session: Session) -> UUID:
    file_bytes = await file.read()
    image_id = uuid4()
    save_path = os.path.join(_get_save_image_directory(), str(image_id))
    Path(save_path).write_bytes(file_bytes)

    db_user = session.get(User, user_id)
    if db_user is None:
        db_user = User(id=user_id)
    db_image = Image(
        id=image_id,
        created_at=datetime.datetime.now(DEFAULT_TIMEZONE),
        name=file.filename,
        mime_type=file.content_type,
        save_path=save_path,
        user=db_user
    )
    session.add(db_image)
    session.commit()

    return image_id


def assign_labels_to_image(image_id: UUID, label_ids: list[int], session: Session):
    db_image = get_image(image_id, session)
    for label_id in label_ids:
        db_label = get_label(label_id, session)
        db_labeled_image = LabeledImage(
            label=db_label,
            image=db_image,
            created_at=datetime.datetime.now(DEFAULT_TIMEZONE)
        )
        session.add(db_labeled_image)
    session.commit()


def delete_image(image_id: UUID, session: Session):
    db_image = get_image(image_id, session)
    session.delete(db_image)
    session.commit()


router = APIRouter(
    prefix="/api/v1/images",
    tags=["Images"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/{image_id}/show", status_code=status.HTTP_200_OK)
async def show(image_id: str, session: SessionDep):
    db_image = get_image(image_id=strict_uuid_parser(image_id), session=session)
    return FileResponse(
        path=db_image.save_path,
        media_type=db_image.mime_type,
        filename=db_image.name
    )


@router.get("/{image_id}/info", response_model=ImagePublic, status_code=status.HTTP_200_OK)
async def get_information(image_id: str, session: SessionDep):
    image_uuid = strict_uuid_parser(image_id)
    return get_image(image_id=image_uuid, session=session)


@router.get("/{label_id}/label", response_model=list[ImagePublic], status_code=status.HTTP_200_OK)
async def get_by_label_id(label_id: int, params: PagingQuery, session: SessionDep):
    return get_images_by_label_id(label_id=label_id, params=params, session=session)


@router.get("/labels", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
async def get_by_label_ids(params: Annotated[LabelsWithPagingParams, Query()], session: SessionDep):
    return get_images_by_label_ids(params=params, session=session)


@router.get("/unlabeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
async def get_unlabeled(params: PagingQuery, session: SessionDep):
    return get_unlabeled_images(params=params, session=session)


@router.get("/labeled", response_model=PagingWrapper[ImagePublic], status_code=status.HTTP_200_OK)
async def get_labeled(params: PagingQuery, session: SessionDep):
    return get_labeled_images(params=params, session=session)


@router.post("/{user_id}/upload", status_code=status.HTTP_201_CREATED)
async def upload(user_id: str, file: UploadFile, session: SessionDep) -> str:
    uploaded_image_id = await save_image(user_id=strict_uuid_parser(user_id), file=file, session=session)
    return str(uploaded_image_id)


@router.post("/{image_id}/assign", status_code=status.HTTP_204_NO_CONTENT)
async def assign_label(image_id: str, label_ids: list[int], session: SessionDep) -> None:
    image_uuid = strict_uuid_parser(image_id)
    assign_labels_to_image(session=session, image_id=image_uuid, label_ids=label_ids)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(image_id: str, session: SessionDep) -> None:
    image_uuid = strict_uuid_parser(image_id)
    delete_image(image_id=image_uuid, session=session)
