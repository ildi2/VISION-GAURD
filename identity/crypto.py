
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception as exc:
    AESGCM = None
    _import_error = exc
else:
    _import_error = None

logger = logging.getLogger(__name__)

_MAGIC = b"GGFG"
_VERSION = 1
_NONCE_LEN = 12
_KEY_LEN = 32

_HEADER_LEN = len(_MAGIC) + 1 + _NONCE_LEN


class CryptoError(Exception):


class CryptoBackendError(CryptoError):


class CryptoKeyError(CryptoError):


class CryptoCiphertextError(CryptoError):


class CryptoVersionError(CryptoCiphertextError):


@dataclass(frozen=True)
class CryptoConfig:
    env_var: str = "GAITGUARD_FACE_KEY"


_key_cache: Dict[str, bytes] = {}
_key_fingerprints: Dict[str, str] = {}


def _ensure_backend_available() -> None:
    if AESGCM is None:
        raise CryptoBackendError(
            "cryptography AESGCM backend not available. "
            "Install the 'cryptography' package. "
            f"Original error: {_import_error}"
        )


def _key_fingerprint(key: bytes) -> str:
    import hashlib

    h = hashlib.sha256(key).hexdigest()
    return h[:8]


def get_cached_key_fingerprint(env_var: str = CryptoConfig.env_var) -> Optional[str]:
    return _key_fingerprints.get(env_var)


def _decode_key(raw: str) -> bytes:
    s = raw.strip()
    key: bytes

    if len(s) in (32, 48, 64) and all(c in "0123456789abcdefABCDEF" for c in s):
        try:
            key = bytes.fromhex(s)
        except ValueError:
            key = b""
    else:
        key = b""

    if not key:
        try:
            key = base64.b64decode(s, validate=True)
        except Exception:
            key = s.encode("utf-8", errors="strict")

    if len(key) not in (16, 24, 32):
        raise CryptoKeyError(
            f"Invalid AES key length {len(key)} bytes; expected 16/24/32."
        )

    if len(key) != _KEY_LEN:
        import hashlib

        key = hashlib.sha256(key).digest()

    return key


def load_key_from_env(env_var: str = CryptoConfig.env_var) -> bytes:
    if env_var in _key_cache:
        return _key_cache[env_var]

    raw = os.getenv(env_var)
    if not raw:
        raise CryptoKeyError(
            f"Encryption key environment variable '{env_var}' is not set. "
            "Generate a strong random key and set it, e.g.:\n"
            "  export GAITGUARD_FACE_KEY=\"$(openssl rand -hex 32)\""
        )

    try:
        key = _decode_key(raw)
    except CryptoKeyError:
        raise
    except Exception as exc:
        raise CryptoKeyError(
            f"Failed to decode AES key from env var '{env_var}': {exc}"
        ) from exc

    _key_cache[env_var] = key
    fp = _key_fingerprint(key)
    _key_fingerprints[env_var] = fp
    logger.info(
        "Loaded AES key from env var '%s' (length=%d bytes, fp=%s).",
        env_var,
        len(key),
        fp,
    )
    return key


def encrypt_bytes(
    plaintext: bytes,
    *,
    key: Optional[bytes] = None,
    env_var: str = CryptoConfig.env_var,
    aad: Optional[bytes] = None,
) -> bytes:
    _ensure_backend_available()

    if not isinstance(plaintext, (bytes, bytearray)):
        raise TypeError("encrypt_bytes expects 'bytes' plaintext.")

    if key is None:
        key = load_key_from_env(env_var)

    if aad is None:
        aad = b""

    nonce = os.urandom(_NONCE_LEN)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext, aad)

    header = _MAGIC + bytes([_VERSION]) + nonce
    return header + ct


def decrypt_bytes(
    blob: bytes,
    *,
    key: Optional[bytes] = None,
    env_var: str = CryptoConfig.env_var,
    aad: Optional[bytes] = None,
) -> bytes:
    _ensure_backend_available()

    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError("decrypt_bytes expects 'bytes' blob.")

    if len(blob) < _HEADER_LEN + 16:
        raise CryptoCiphertextError("Ciphertext too short to be valid.")

    if key is None:
        key = load_key_from_env(env_var)

    if aad is None:
        aad = b""

    if blob[: len(_MAGIC)] != _MAGIC:
        raise CryptoVersionError("Invalid ciphertext magic header.")

    version = blob[len(_MAGIC)]
    if version != _VERSION:
        raise CryptoVersionError(
            f"Unsupported ciphertext version {version}; "
            f"this code expects version={_VERSION}."
        )

    offset = len(_MAGIC) + 1
    nonce = blob[offset : offset + _NONCE_LEN]
    ct = blob[offset + _NONCE_LEN :]

    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ct, aad)
    except Exception as exc:
        fp = _key_fingerprint(key)
        raise CryptoCiphertextError(
            f"Decryption failed or authentication tag invalid "
            f"(possible key mismatch or corrupted file; key_fp={fp}): {exc}"
        ) from exc


def encrypt_json(
    obj: Any,
    *,
    key: Optional[bytes] = None,
    env_var: str = CryptoConfig.env_var,
    aad: Optional[bytes] = None,
    ensure_ascii: bool = False,
) -> bytes:
    data = json.dumps(
        obj,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
    ).encode("utf-8")
    return encrypt_bytes(data, key=key, env_var=env_var, aad=aad)


def decrypt_json(
    blob: bytes,
    *,
    key: Optional[bytes] = None,
    env_var: str = CryptoConfig.env_var,
    aad: Optional[bytes] = None,
) -> Any:
    data = decrypt_bytes(blob, key=key, env_var=env_var, aad=aad)
    try:
        return json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise CryptoCiphertextError(
            f"Decrypted data is not valid JSON: {exc}"
        ) from exc
