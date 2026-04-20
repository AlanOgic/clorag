"""Encrypted OAuth token storage utilities.

Provides secure storage for OAuth tokens using Fernet symmetric encryption.
The encryption key is derived from the admin password using PBKDF2.
"""

import base64
import json
import os
import shutil
from datetime import date
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from clorag.config import get_settings
from clorag.utils.logger import get_logger

logger = get_logger(__name__)

def _get_salt_path() -> Path:
    """Resolve salt file path from database_path for consistency across working dirs."""
    settings = get_settings()
    return Path(settings.database_path).parent / ".token_salt"


def _get_or_create_salt() -> bytes:
    """Get or create a salt for key derivation.

    The salt is stored in a separate file and should be backed up
    along with the encrypted tokens.
    """
    salt_path = _get_salt_path()
    if salt_path.exists():
        return salt_path.read_bytes()

    salt = os.urandom(16)
    salt_path.parent.mkdir(parents=True, exist_ok=True)
    salt_path.write_bytes(salt)
    return salt


def _derive_key(password: str) -> bytes:
    """Derive a Fernet key from password using PBKDF2."""
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
    """Get a Fernet instance for token encryption.

    Uses token_encryption_key if set, otherwise falls back to admin_password.
    """
    settings = get_settings()
    if settings.token_encryption_key:
        key = _derive_key(settings.token_encryption_key.get_secret_value())
        return Fernet(key)

    if settings.admin_password:
        key = _derive_key(settings.admin_password.get_secret_value())
        return Fernet(key)

    logger.warning("No encryption key configured, token encryption disabled")
    return None


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


def _plaintext_fallback_allowed() -> bool:
    """Return True when the configured cutoff has not yet elapsed.

    Empty/None cutoff disables the time-bound guard (legacy behavior);
    this must be set explicitly by an operator.
    """
    cutoff_str = get_settings().token_plaintext_cutoff
    if not cutoff_str:
        return True
    try:
        cutoff = date.fromisoformat(cutoff_str)
    except ValueError:
        logger.warning(
            "Invalid TOKEN_PLAINTEXT_CUTOFF, treating as unset",
            value=cutoff_str,
        )
        return True
    return date.today() < cutoff


def decrypt_token_data(encrypted_data: str) -> dict[str, object] | None:
    """Decrypt token data from storage.

    Args:
        encrypted_data: Base64-encoded encrypted data or plain JSON.

    Returns:
        Decrypted token data dictionary, or None if decryption fails.
    """
    # Try plain JSON first (bounded backwards compatibility)
    try:
        data = json.loads(encrypted_data)
        if isinstance(data, dict):
            if any(k in data for k in ["token", "refresh_token", "client_id"]):
                if not _plaintext_fallback_allowed():
                    logger.error(
                        "Refusing to load plaintext OAuth token past"
                        " TOKEN_PLAINTEXT_CUTOFF; rotate the credential.",
                    )
                    return None
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

    If the file is still stored as plaintext JSON (legacy layout) and the
    plaintext fallback is still within its cutoff, the token is re-encrypted
    in place so subsequent loads never touch plaintext again.

    Args:
        token_path: Path to the encrypted token file.

    Returns:
        Decrypted token data dictionary, or None if file doesn't exist or
        decryption fails / plaintext was rejected past the cutoff.
    """
    if not token_path.exists():
        return None

    raw = token_path.read_text()
    data = decrypt_token_data(raw)
    if data is None:
        return None

    # If we just read plaintext JSON, upgrade the file in place (best-effort).
    try:
        parsed = json.loads(raw)
        is_plaintext = isinstance(parsed, dict) and any(
            k in parsed for k in ["token", "refresh_token", "client_id"]
        )
    except json.JSONDecodeError:
        is_plaintext = False

    if is_plaintext and _get_fernet() is not None:
        try:
            save_encrypted_token(token_path, data)
            logger.info(
                "Migrated plaintext OAuth token to encrypted storage",
                path=str(token_path),
            )
        except Exception as e:
            # Never block callers on migration failure; the raw plaintext is
            # still usable (and will retry on next load).
            logger.warning(
                "Plaintext token migration failed",
                path=str(token_path),
                error=str(e),
            )
    return data
