import logging
from pathlib import Path

from .interface.label import ILabelService
from ..data.base_model import LabelSource
from ..data.model import Label
from ..process.recognizer import RecognizerOutput
from ..repository.interface.label import ILabelRepository
from ..util.error import NotFoundError, InvalidArgumentError


class LabelServiceImpl(ILabelService):
    _label_repository: ILabelRepository
    _logger = logging.getLogger(__name__)

    def __init__(self, label_repository: ILabelRepository):
        super().__init__()
        self._label_repository = label_repository

    async def get_all_labels(self):
        return await self._label_repository.get_all()

    async def get_label_by_id(self, label_id):
        db_label = await self._label_repository.get_by_id(entity_id=label_id)
        if db_label is None:
            raise NotFoundError(f'No label with id {label_id} found.')
        return db_label

    async def get_labels_by_image_id(self, image_id):
        return await self._label_repository.get_all_by_image_id(image_id)

    async def create_label(self, label):
        # Check exist label name
        exist_label = await self._label_repository.get_by_name(label.name)
        if exist_label is not None:
            raise InvalidArgumentError(f'Label with name {label.name} already exists.')

        db_label = await self._label_repository.save(Label(name=label.name,
                                                           description=label.description,
                                                           source=LabelSource.CREATED))
        return db_label

    async def update_label(self, label_id, label):
        db_label = await self.get_label_by_id(label_id)
        db_label.description = label.description
        await self._label_repository.save(db_label)

    async def delete_label_by_name(self, label_name):
        deleted_label = await self._label_repository.get_by_name(label_name)
        if deleted_label is None:
            raise NotFoundError(f'No label with name {label_name} found.')
        return deleted_label

    async def delete_label_by_id(self, label_id):
        deleted_label = await self._label_repository.delete_by_id(label_id)
        if deleted_label is None:
            raise NotFoundError(f'No label with id {label_id} found.')
        return deleted_label

    async def insert_predefined_output_classes(self, config_file_path):
        self._logger.debug(f"Reading and validating predefined output classes from config file: {config_file_path}")
        file_path = Path(config_file_path)
        json_bytes = file_path.read_bytes()
        output = RecognizerOutput.model_validate_json(json_bytes)
        if output.is_configured:
            self._logger.debug("Predefined output classes are already configured. Skipping...")
            return

        self._logger.debug(f'Saving predefined output classes to database...')
        db_labels = [Label(name=output_class.name,
                           description=output_class.description,
                           source=LabelSource.PREDEFINED) for output_class in output.classes]
        await self._label_repository.save_all(db_labels)

        classes = "\n".join(
            f'name: {output_class.name}, desc: {output_class.description}'
            for output_class in output.classes)
        self._logger.debug(f"Labels are saved to database:\n{classes}")

        self._logger.debug(f'Updating config file with configured status: is_configured = True...')
        with open(config_file_path, 'w'):  # Clear old content
            pass
        output.is_configured = True  # Mark as configured
        file_path.write_text(output.model_dump_json(indent=2))
