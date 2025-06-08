import logging
import os

import jsonpickle

from src.config.model.main import AgentConfiguration
from src.process.recognizer.image.main import ImageRecognizer
from src.util.function import get_config_folder_path


def _get_config_file_path():
    config_file_name = "config.json"
    config_path = os.path.join(get_config_folder_path(), config_file_name)
    if os.path.exists(config_path) is False:
        raise FileNotFoundError(f'Missing {config_file_name} file in {config_path}')
    return config_path


class AgentConfigurer:
    _config: AgentConfiguration | None = None
    _image_recognizer: ImageRecognizer | None = None
    _logger = logging.getLogger(__name__)

    def _load_config(self):
        """
        Loads the agent configuration from the configuration file.

        This method reads the JSON content from the config file and
        validates it against the `AgentConfiguration` Pydantic model.
        The loaded configuration is then stored in the `self._config` attribute.

        Raises:
            FileNotFoundError: If the `DEFAULT_CONFIG_PATH` does not exist.
            pydantic.ValidationError: If the content of the configuration file
                does not conform to the structure defined by the `AgentConfiguration` model.
            Exception: For other potential errors during file reading.

        Returns:
            None
        """
        config_file_path = _get_config_file_path()
        self._logger.info(f'Loading configuration...')
        with open(config_file_path, mode="r") as config_file:
            json = config_file.read()
        self._config = AgentConfiguration.model_validate(jsonpickle.decode(json))
        self._logger.info(f'Loaded configuration successfully!')

    def configure(self):
        self._load_config()

    @property
    def config(self):
        return self._config
