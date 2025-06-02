import datetime
import os.path
import typing
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, UploadFile, status
from sqlmodel import Session, select

from .label import get_label
from ..data.dto import ImagePublic
from ..data.model import Image, User, LabeledImage
from ..dependency import SessionDep, DownloadGeneratorDep, PagingQuery, PagingParams
from ..util.error import NotFoundError
from ..util.main import strict_uuid_parser, SecureDownloadGenerator, FileInformation, DEFAULT_TIMEZONE

DEFAULT_SAVE_DIRECTORY = "/resource"
save_image_directory = os.getenv("SAVE_IMAGE_DIRECTORY", DEFAULT_SAVE_DIRECTORY)


def get_image(image_id: UUID, session: Session) -> Image:
    db_image = session.get(Image, image_id)
    if db_image is None:
        raise NotFoundError(f'Image with id {image_id} not found.')
    return typing.cast(Image, db_image)


def get_image_download_token(image_id: UUID, session: Session, generator: SecureDownloadGenerator) -> str:
    db_image = get_image(image_id, session)  # check image existence
    data: FileInformation = {
        "name": db_image.name,
        "mime_type": db_image.mime_type,
        "path": db_image.save_path,
    }
    return generator.generate_token(data=data)


# noinspection PyTypeChecker,PyComparisonWithNone
def get_images_by_label_id(label_id: UUID, params: PagingParams, session: Session) -> list[Image]:
    statement = (select(Image)
                 .join(LabeledImage, LabeledImage.image_id == Image.id)
                 .where(LabeledImage.label_id == label_id)
                 .offset(params.offset)
                 .limit(params.limit)
                 .order_by(LabeledImage.created_at))
    results = session.exec(statement)
    return list(results.all())


# noinspection PyTypeChecker,PyComparisonWithNone
def get_unlabeled_images(params: PagingParams, session: Session) -> list[Image]:
    statement = (select(Image)
                 .join(LabeledImage, LabeledImage.image_id == Image.id, isouter=True)
                 .where(LabeledImage.label_id == None)
                 .offset(params.offset)
                 .limit(params.limit)
                 .order_by(LabeledImage.created_at))
    results = session.exec(statement)
    return list(results.all())


async def save_image(user_id: UUID, file: UploadFile, session: Session) -> UUID:
    file_bytes = await file.read()
    image_id = uuid4()
    save_path = os.path.join(save_image_directory, str(image_id))
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


def assign_label_to_image(image_id: UUID, label_id: int, session: Session):
    db_image = get_image(image_id, session)
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


router = APIRouter(
    prefix="/api/v1/images",
    tags=["Images"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/{image_id}/token", status_code=status.HTTP_200_OK)
async def get_download_token(image_id: str, session: SessionDep,
                             generator: DownloadGeneratorDep) -> str:
    image_uuid = strict_uuid_parser(image_id)
    return get_image_download_token(image_id=image_uuid, session=session, generator=generator)


@router.get("/{image_id}/info", response_model=ImagePublic, status_code=status.HTTP_200_OK)
async def get_information(image_id: str, session: SessionDep):
    image_uuid = strict_uuid_parser(image_id)
    return get_image(image_id=image_uuid, session=session)


@router.get("/{label_id}/label", response_model=list[ImagePublic], status_code=status.HTTP_200_OK)
async def get_by_label_id(label_id: str, params: PagingQuery, session: SessionDep):
    return get_images_by_label_id(label_id=strict_uuid_parser(label_id), params=params, session=session)


@router.get("/unlabeled", response_model=list[ImagePublic], status_code=status.HTTP_200_OK)
async def get_unlabeled(params: PagingQuery, session: SessionDep):
    return get_unlabeled_images(params=params, session=session)


@router.post("/{user_id}/upload", status_code=status.HTTP_201_CREATED)
async def upload(user_id: str, file: UploadFile, session: SessionDep) -> str:
    uploaded_image_id = await save_image(user_id=strict_uuid_parser(user_id), file=file, session=session)
    return str(uploaded_image_id)


@router.post("/{image_id}/assign/{label_id}", status_code=status.HTTP_204_NO_CONTENT)
async def assign_label(image_id: str, label_id: int, session: SessionDep) -> None:
    image_uuid = strict_uuid_parser(image_id)
    assign_label_to_image(session=session, image_id=image_uuid, label_id=label_id)


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(image_id: str, session: SessionDep) -> None:
    image_uuid = strict_uuid_parser(image_id)
    delete_image(image_id=image_uuid, session=session)
