import base64
import hashlib
import hmac
import secrets
import time
import uuid

from src.error import InvalidArgumentError


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


DEFAULT_CHARSET = "utf-8"


class SecureDownloadGenerator:
    """Generate secure, time-limited download links"""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode(DEFAULT_CHARSET)

    def generate_token(self, file_id: str, expires_in: int = 3600, user_id: str | None = None) -> str:
        """Generate a secure token for file download."""
        expiry = int(time.time()) + expires_in
        nonce = secrets.token_urlsafe(16)

        # Include user_id in the payload if provided
        payload_parts = [file_id, str(expiry), nonce]
        if user_id:
            payload_parts.append(user_id)
        payload = ":".join(payload_parts)

        # Create signature
        signature = hmac.new(
            self.secret_key,
            payload.encode(DEFAULT_CHARSET),
            hashlib.sha256
        ).hexdigest()

        # Combine payload and signature
        token_data = f"{payload}:{signature}"

        # Base64 encode for URL safety
        token = base64.urlsafe_b64encode(token_data.encode(DEFAULT_CHARSET)).decode(DEFAULT_CHARSET)
        return token

    def verify_token(self, token: str) -> str | None:
        """Verify a download token and return a file id."""
        # Decode base64
        token_data = base64.urlsafe_b64decode(token.encode(DEFAULT_CHARSET)).decode(DEFAULT_CHARSET)

        # Split token parts
        parts = token_data.split(':')
        if len(parts) < 4:
            return None

        # Extract parts
        file_id = parts[0]
        expiry_str = parts[1]
        nonce = parts[2]

        # Check if user_id is included
        if len(parts) == 5:
            user_id = parts[3]
            signature = parts[4]
            payload = f"{file_id}:{expiry_str}:{nonce}:{user_id}"
        else:
            signature = parts[3]
            payload = f"{file_id}:{expiry_str}:{nonce}"

        # Check expiration
        expiry = int(expiry_str)
        if time.time() > expiry:
            return None

        # Verify signature
        expected_signature = hmac.new(
            self.secret_key,
            payload.encode(DEFAULT_CHARSET),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return None

        return file_id
