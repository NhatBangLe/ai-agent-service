import typing

from fastapi import APIRouter
from sqlmodel import Session, select

from ..dependency import SessionDep
from ..data.dto import LabelPublic, LabelCreate
from ..data.model import Label
from ..util.error import NotFoundError


def get_label(label_id: int, session: Session):
    db_label = session.get(Label, label_id)
    if db_label is None:
        raise NotFoundError(f'Label with id {label_id} not found.')
    return typing.cast(Label, db_label)


def create_label(session: Session, label: LabelCreate):
    db_label = Label.model_validate(label)
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
