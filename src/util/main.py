import base64
import hashlib
import hmac
import secrets
import time
from os import PathLike
from typing import TypedDict

from src.util.constant import DEFAULT_CHARSET, DEFAULT_TOKEN_SEPARATOR


class FileInformation(TypedDict):
    """File information dictionary"""
    name: str
    mime_type: str
    path: str | PathLike[str]


class SecureDownloadGenerator:
    """Generate secure, time-limited download links"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode(DEFAULT_CHARSET)

    def generate_token(self, data: FileInformation, expires_in: int = 3600, user_id: str | None = None) -> str:
        """Generate a secure token for file download."""
        expiry = int(time.time()) + expires_in
        nonce = secrets.token_urlsafe(16)

        # Include user_id in the payload if provided
        payload_parts = [data["name"], str(data["path"]), data["mime_type"], str(expiry), nonce]
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


class Progress(TypedDict):
    """
    A dictionary representing the progress of an operation.
    """
    status: str
    percentage: float
