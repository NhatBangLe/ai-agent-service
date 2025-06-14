import base64
import hashlib
import hmac
import re
import secrets
import time

from os import PathLike
from pathlib import Path
from typing import TypedDict

from src.util.constant import DEFAULT_CHARSET, DEFAULT_TOKEN_SEPARATOR, EMOTICONS

__all__ = ['error', 'FileInformation', "SecureDownloadGenerator", "Progress", "constant", "function"]


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


class TextPreprocessing:
    _removal_words: list[str]

    def __init__(self, removal_words_path: str | PathLike[str]):
        super().__init__()
        all_words = Path(removal_words_path).read_text(encoding=DEFAULT_CHARSET)
        self._removal_words = all_words.split('\n')

    @staticmethod
    def remove_emoji(text: str) -> str:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    @staticmethod
    def remove_emoticons(text):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        return emoticon_pattern.sub(r'', text)

    def remove_words(self, text: str) -> str:
        return " ".join([word for word in str(text).split() if word not in self._removal_words])
