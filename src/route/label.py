import typing
from uuid import UUID

from fastapi import APIRouter, status
from sqlmodel import Session, select

from ..data.base_model import LabelSource
from ..data.dto import LabelPublic, LabelCreate
from ..data.model import Label, LabeledImage
from ..dependency import SessionDep, PagingQuery
from ..util import PagingParams
from ..util.error import NotFoundError
from ..util.function import strict_uuid_parser


def get_label(label_id: int, session: Session):
    db_label = session.get(Label, label_id)
    if db_label is None:
        raise NotFoundError(f'Label with id {label_id} not found.')
    return typing.cast(Label, db_label)


# noinspection PyTypeChecker
def get_labels_by_image_id(image_id: UUID, params: PagingParams, session: Session) -> list[Label]:
    statement = (select(Label)
                 .join(LabeledImage, LabeledImage.label_id == Label.id)
                 .where(LabeledImage.image_id == image_id)
                 .order_by(LabeledImage.created_at)
                 .offset(params.offset)
                 .limit(params.limit))
    results = session.exec(statement)
    return list(results.all())


def create_label(label: LabelCreate, session: Session):
    db_label = Label(name=label.name,
                     description=label.description,
                     source=LabelSource.CREATED)
    session.add(db_label)
    session.commit()
    session.refresh(db_label)
    return db_label


def read_labels(session: Session):
    statement = select(Label)
    results = session.exec(statement)
    return list(results)


router = APIRouter(
    prefix="/api/v1/labels",
    tags=["Labels"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/all", response_model=list[LabelPublic])
async def get_labels(session: SessionDep):
    return read_labels(session)


@router.get("/{image_id}/image", response_model=list[LabelPublic], status_code=status.HTTP_200_OK)
async def get_by_image_id(image_id: str, params: PagingQuery, session: SessionDep):
    return get_labels_by_image_id(image_id=strict_uuid_parser(image_id), params=params, session=session)


@router.post("/create", response_model=LabelPublic, status_code=status.HTTP_201_CREATED)
async def create(label: LabelCreate, session: SessionDep):
    return create_label(session=session, label=label)
