from dependency_injector.wiring import inject
from fastapi import APIRouter, status

from ..dependency import DownloadGeneratorDepend
from ..dependency import ExportingServiceDepend

router = APIRouter(
    prefix="/api/v1/export",
    tags=["Export"],
    responses={
        400: {"description": "Invalid parameter(s)."},
        404: {"description": "Entity not found."}
    },
)


@router.get("/all", status_code=status.HTTP_200_OK)
@inject
async def get_exporting_all_token(service: ExportingServiceDepend, generator: DownloadGeneratorDepend):
    metadata = await service.export_all_labeled_images()
    return generator.generate_token({
        "name": metadata.name,
        "mime_type": metadata.mime_type,
        "path": metadata.path
    })


@router.get("/{label_id}/label", status_code=status.HTTP_200_OK)
@inject
async def export_by_label_id(label_id: int, service: ExportingServiceDepend, generator: DownloadGeneratorDepend):
    metadata = await service.export_labeled_images_by_label_id(label_id)
    return generator.generate_token({
        "name": metadata.name,
        "mime_type": metadata.mime_type,
        "path": metadata.path
    })
