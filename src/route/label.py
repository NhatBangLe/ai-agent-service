from dependency_injector.wiring import inject
from fastapi import APIRouter, status

from ..data.dto import LabelPublic, LabelCreate, LabelDelete, LabelUpdate
from ..dependency import LabelServiceDepend
from ..util.error import InvalidArgumentError
from ..util.function import strict_uuid_parser

router = APIRouter(
    prefix="/api/v1/labels",
    tags=["Labels"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/all", response_model=list[LabelPublic], status_code=status.HTTP_200_OK)
@inject
async def get_labels(service: LabelServiceDepend):
    return await service.get_all_labels()


@router.get("/{image_id}/image", response_model=list[LabelPublic], status_code=status.HTTP_200_OK)
@inject
async def get_by_image_id(image_id: str, service: LabelServiceDepend):
    image_uuid = strict_uuid_parser(image_id)
    return await service.get_labels_by_image_id(image_id=image_uuid)


@router.get("/{label_id}", response_model=LabelPublic, status_code=status.HTTP_200_OK)
@inject
async def get_by_label_id(label_id: int, service: LabelServiceDepend):
    return await service.get_label_by_id(label_id=label_id)


@router.post("/create", status_code=status.HTTP_201_CREATED)
@inject
async def create(label: LabelCreate, service: LabelServiceDepend) -> int:
    created_label = await service.create_label(label)
    return created_label.id


@router.put("/{label_id}/update", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def update(label_id: int, label: LabelUpdate, service: LabelServiceDepend):
    await service.update_label(label_id, label)


@router.delete("/delete", status_code=status.HTTP_204_NO_CONTENT)
@inject
async def delete(params: LabelDelete, service: LabelServiceDepend):
    label_id = params.id
    name = params.name
    if label_id is not None:
        await service.delete_label_by_id(label_id)
    elif name is not None:
        await service.delete_label_by_name(name)
    else:
        raise InvalidArgumentError(f'Must specify id or name of label to delete.')
