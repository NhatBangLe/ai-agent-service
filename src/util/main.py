import base64
import hashlib
import hmac
import secrets
import time
import uuid
import datetime
from typing import TypedDict

from src.util.error import InvalidArgumentError


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


DEFAULT_TIMEZONE = datetime.timezone.utc


def convert_datetime_to_str(datetime_obj: datetime.datetime) -> str:
    """
    Convert a datetime object to string.
    `DEFAULT_TIMEZONE` is used as the timezone.
    """
    return datetime_obj.astimezone(DEFAULT_TIMEZONE).isoformat()


def convert_str_to_datetime(datetime_str: str) -> datetime.datetime:
    """
    Convert a string to a datetime object.
    The `datetime_str` must be in ISO 8601 format.
    `DEFAULT_TIMEZONE` is used as the timezone.

    Args:
        datetime_str: String representation of a datetime object

    Raises:
        ValueError: If datetime string is invalid
    """
    return datetime.datetime.fromisoformat(datetime_str).astimezone(DEFAULT_TIMEZONE)


DEFAULT_CHARSET = "utf-8"
DEFAULT_TOKEN_SEPARATOR = "::"


class FileInformation(TypedDict):
    """File information dictionary"""
    name: str
    mime_type: str | None
    path: str


class SecureDownloadGenerator:
    """Generate secure, time-limited download links"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode(DEFAULT_CHARSET)

    def generate_token(self, data: FileInformation, expires_in: int = 3600, user_id: str | None = None) -> str:
        """Generate a secure token for file download."""
        expiry = int(time.time()) + expires_in
        nonce = secrets.token_urlsafe(16)

        # Include user_id in the payload if provided
        payload_parts = [data["name"], data["path"], data["mime_type"], str(expiry), nonce]
        if user_id:
            payload_parts.append(user_id)
        payload = DEFAULT_TOKEN_SEPARATOR.join(payload_parts)

        # Create signature
        signature = hmac.new(
            self.secret_key,
            payload.encode(DEFAULT_CHARSET),
            hashlib.sha256
        ).hexdigest()

        # Combine payload and signature
        token_data = f"{payload}{DEFAULT_TOKEN_SEPARATOR}{signature}"

        # Base64 encode for URL safety
        token = base64.urlsafe_b64encode(token_data.encode(DEFAULT_CHARSET)).decode(DEFAULT_CHARSET)
        return token

    def verify_token(self, token: str) -> FileInformation | None:
        """Verify a download token and return a file id."""
        # Decode base64
        token_data: str = base64.urlsafe_b64decode(token.encode(DEFAULT_CHARSET)).decode(DEFAULT_CHARSET)

        # Split token parts
        parts: list[str] = token_data.split(DEFAULT_TOKEN_SEPARATOR)
        if len(parts) < 5:
            return None

        # Extract parts
        name, path, mime_type, expiry_str, nonce = parts[:5]

        # Check expiration
        expiry = int(expiry_str)
        if time.time() > expiry:
            return None

        # Check if user_id is included
        if len(parts) == 7:
            user_id = parts[5]
            signature = parts[6]
            payload = DEFAULT_TOKEN_SEPARATOR.join([name, path, mime_type, expiry_str, nonce, user_id])
        else:
            signature = parts[5]
            payload = DEFAULT_TOKEN_SEPARATOR.join([name, path, mime_type, expiry_str, nonce])

        # Verify signature
        expected_signature = hmac.new(
            self.secret_key,
            payload.encode(DEFAULT_CHARSET),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return None

        return FileInformation(
            name=name,
            mime_type=mime_type,
            path=path,
        )
