import datetime
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .constant import EnvVar, DEFAULT_TIMEZONE
from ..util.error import InvalidArgumentError


def get_cache_dir_path():
    return Path(os.getenv(EnvVar.CACHE_DIR, "/resource_cache"))


def get_datetime_now():
    return datetime.datetime.now(DEFAULT_TIMEZONE)


def get_config_folder_path():
    config_path = os.getenv(EnvVar.AGENT_CONFIG_DIR.value)
    if config_path is None:
        raise RuntimeError(f"Missing the {EnvVar.AGENT_CONFIG_DIR.value} environment variable.")
    return config_path


def strict_uuid_parser(uuid_string: str) -> uuid.UUID:
    """
    Strict UUID parser that raises an exception on invalid input.

    Args:
        uuid_string: String representation of UUID

    Returns:
        uuid.UUID object

    Raises:
        InvalidArgumentError: If UUID string is invalid
    """
    try:
        return uuid.UUID(uuid_string)
    except (ValueError, TypeError) as e:
        raise InvalidArgumentError(f"Invalid UUID format: {uuid_string}") from e


def shrink_file_name(max_name_len: int, file_name: str, ext: str | None = None) -> str:
    """
    Shortens the provided file name to a specified maximum length. If the given
    file name is longer than the allowed length, the function truncates it and
    appends the specified or derived file extension. If no file name is provided,
    a default name based on the current datetime is generated.

    :param max_name_len: Maximum allowed length for the resulting file name.
    :param file_name: Original file name to be shortened.
    :param ext: Optional. File extension to enforce (must include dot); if not provided, the
        extension is derived from the original file name.
    :return: A valid file name string within the specified maximum length.
    """
    filename_arr = file_name.split('.')
    if ext is None:
        ext = f'.{filename_arr[-1]}'
    if len(file_name) > max_name_len:
        file_name = filename_arr[0]
        max_len_value = max_name_len - len(ext)
        if len(file_name) > max_len_value:
            file_name = file_name[:max_len_value]
    return file_name


def is_web_path(path: str) -> bool:
    """
    Checks if a given string is likely a web URL.
    """
    try:
        result = urlparse(path)
        # A web URL usually has a scheme (http, https, ftp) and a network location.
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
