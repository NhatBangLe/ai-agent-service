from fastapi import APIRouter, status
from sqlmodel import Session, select

from ..dependency import SessionDep, DownloadGeneratorDep, PagingParams
from ..data.model import Image, LabeledImage, Label
from ..util.error import NotFoundError
from ..util.main import SecureDownloadGenerator

DEFAULT_CACHE_DIRECTORY = "/resource"


def get_exporting_labeled_images_token(label_id: int, session: Session, generator: SecureDownloadGenerator) -> str:
    db_label = session.get(Label, label_id)
    if db_label is None:
        raise NotFoundError(f'No label with id {label_id} found.')
    return ''


def get_exporting_all_labeled_images_token(session: Session, generator: SecureDownloadGenerator) -> str:
    return ''


# noinspection PyTypeChecker,PyComparisonWithNone
def get_unlabeled_images(params: PagingParams, session: Session) -> list[Image]:
    statement = (select(Image)
                 .join(LabeledImage, isouter=True)
                 .where(LabeledImage.label_id == None)
                 .offset(params.offset)
                 .limit(params.limit))
    results = session.exec(statement)
    return list(results.all())


router = APIRouter(
    prefix="/route/v1/export",
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
