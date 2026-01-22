"""Encrypted OAuth token storage utilities.

Provides secure storage for OAuth tokens using Fernet symmetric encryption.
The encryption key is derived from the admin password using PBKDF2.
"""

import base64
import json
import os
import shutil
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from clorag.config import get_settings
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

# Salt file path (stored separately from encrypted tokens)
SALT_FILE = Path("data/.token_salt")


def _get_or_create_salt() -> bytes:
    """Get or create a salt for key derivation.

    The salt is stored in a separate file and should be backed up
    along with the encrypted tokens.
    """
    if SALT_FILE.exists():
        return SALT_FILE.read_bytes()

    # Create a new random salt
    salt = os.urandom(16)
    SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SALT_FILE.write_bytes(salt)
    return salt


def _derive_key(password: str) -> bytes:
    """Derive a Fernet key from the admin password using PBKDF2."""
    salt = _get_or_create_salt()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,  # OWASP recommended minimum
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def _get_fernet() -> Fernet | None:
    """Get a Fernet instance using the admin password as key.

    Returns None if admin password is not configured.
    """
    settings = get_settings()
    if not settings.admin_password:
        logger.warning("Admin password not configured, token encryption disabled")
        return None

    key = _derive_key(settings.admin_password.get_secret_value())
    return Fernet(key)


def encrypt_token_data(data: dict[str, object]) -> str:
    """Encrypt token data for secure storage.

    Args:
        data: Token data dictionary to encrypt.

    Returns:
        Base64-encoded encrypted data.

    Raises:
        RuntimeError: If ADMIN_PASSWORD is not configured (encryption unavailable).
    """
    fernet = _get_fernet()
    if fernet is None:
        # SECURITY: Never fall back to plaintext - OAuth tokens must be encrypted
        raise RuntimeError(
            "Cannot encrypt tokens: ADMIN_PASSWORD must be configured. "
            "Set the ADMIN_PASSWORD environment variable to enable secure token storage."
        )

    json_data = json.dumps(data).encode()
    encrypted = fernet.encrypt(json_data)
    return encrypted.decode()


def decrypt_token_data(encrypted_data: str) -> dict[str, object] | None:
    """Decrypt token data from storage.

    Args:
        encrypted_data: Base64-encoded encrypted data or plain JSON.

    Returns:
        Decrypted token data dictionary, or None if decryption fails.
    """
    # Try plain JSON first (for backwards compatibility)
    try:
        data = json.loads(encrypted_data)
        if isinstance(data, dict):
            # Check if this looks like unencrypted token data
            if any(k in data for k in ["token", "refresh_token", "client_id"]):
                logger.info("Found unencrypted token data, will encrypt on next save")
                return data
    except json.JSONDecodeError:
        pass

    # Try decryption
    fernet = _get_fernet()
    if fernet is None:
        logger.warning("Cannot decrypt token: admin password not configured")
        return None

    try:
        decrypted = fernet.decrypt(encrypted_data.encode())
        result: dict[str, object] = json.loads(decrypted.decode())
        return result
    except InvalidToken:
        logger.error("Failed to decrypt token: invalid key or corrupted data")
        return None
    except Exception as e:
        logger.error("Failed to decrypt token", error=str(e))
        return None


def save_encrypted_token(token_path: Path, token_data: dict[str, object]) -> None:
    """Save token data with encryption.

    Args:
        token_path: Path to save the encrypted token file.
        token_data: Token data dictionary to encrypt and save.
    """
    encrypted = encrypt_token_data(token_data)
    token_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file then move for atomic operation
    # Use shutil.move instead of rename to handle Docker volume mounts
    temp_path = token_path.with_suffix(".tmp")
    temp_path.write_text(encrypted)
    shutil.move(str(temp_path), str(token_path))

    # Set restrictive permissions (owner read/write only)
    token_path.chmod(0o600)

    logger.info("Saved encrypted token", path=str(token_path))


def load_encrypted_token(token_path: Path) -> dict[str, object] | None:
    """Load and decrypt token data from file.

    Args:
        token_path: Path to the encrypted token file.

    Returns:
        Decrypted token data dictionary, or None if file doesn't exist or decryption fails.
    """
    if not token_path.exists():
        return None

    encrypted_data = token_path.read_text()
    return decrypt_token_data(encrypted_data)
