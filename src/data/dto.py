from uuid import UUID

from .base_model import BaseImage, BaseLabel


class LabelPublic(BaseLabel):
    id: int


class LabelCreate(BaseLabel):
    pass


class ImagePublic(BaseImage):
    id: UUID
