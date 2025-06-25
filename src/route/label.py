import typing
from uuid import UUID

from fastapi import APIRouter, status
from sqlmodel import Session, select

from ..data.base_model import LabelSource
from ..data.dto import LabelPublic, LabelCreate, LabelDelete, LabelUpdate
from ..data.model import Label, LabeledImage
from ..dependency import SessionDep, PagingQuery
from ..util import PagingParams
from ..util.error import NotFoundError, InvalidArgumentError
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


# noinspection PyTypeChecker
def create_label(label: LabelCreate, session: Session):
    # Check exist label name
    exist_label = (session.exec(select(Label).where(Label.name == label.name).limit(1))
                   .one_or_none())
    if exist_label is not None:
        raise InvalidArgumentError(f'Label with name {label.name} already exists.')

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


# noinspection PyTypeChecker
def update_label(label_id: int, label_update: LabelUpdate, session: Session):
    db_label = get_label(label_id, session)
    db_label.description = label_update.description

    session.add(db_label)
    session.commit()
    return typing.cast(Label, db_label)


# noinspection PyTypeChecker
def delete_label(params: LabelDelete, session: Session):
    if params.id is None and params.name is None:
        raise InvalidArgumentError(f'Must specify id or name of label to delete.')

    if params.id is not None:
        db_label = get_label(params.id, session)
    else:
        db_label = (session.exec(select(Label).where(Label.name == params.name).limit(1))
                    .one_or_none())
        if db_label is None:
            raise NotFoundError(f'No label with {params.name} found.')
    if db_label.source == LabelSource.PREDEFINED:
        raise InvalidArgumentError(f'Cannot delete a predefined label.')

    session.delete(db_label)
    session.commit()


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


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create(label: LabelCreate, session: SessionDep) -> int:
    db_label = create_label(session=session, label=label)
    return db_label.id


@router.put("/{label_id}/update", status_code=status.HTTP_204_NO_CONTENT)
async def update(label_id: int, label: LabelUpdate, session: SessionDep):
    update_label(label_id=label_id, label_update=label, session=session)


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def create(params: LabelDelete, session: SessionDep):
    delete_label(params=params, session=session)
