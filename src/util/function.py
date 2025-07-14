import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

from src.util.error import InvalidArgumentError


def get_config_folder_path():
    config_path = os.getenv("AGENT_CONFIG_PATH")
    if config_path is None:
        raise RuntimeError("Missing the AGENT_CONFIG_PATH environment variable.")
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


# noinspection PyAbstractClass
async def get_documents(file_path: str | bytes, mime_type: str) -> list[Document]:
    if mime_type == "application/pdf":
        loader = PyPDFLoader(file_path)
        documents = await loader.aload()
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file_path)
        documents = await loader.aload()
    elif mime_type == "text/plain":
        text = Path(file_path).read_text(encoding="utf-8")
        documents = [Document(page_content=text, metadata={"source": file_path, "mime_type": "text/plain"})]
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")

    return documents


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
