from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass


_PBKDF2_ITERS = 210_000 # repeatedly hashes password many times.
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """
    PBKDF2-HMAC-SHA256 password hash, no external deps.
    Stored format: pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>
    """
    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return "pbkdf2_sha256$%d$%s$%s" % (
        _PBKDF2_ITERS,
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(dk).decode("ascii").rstrip("="),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = base64.urlsafe_b64decode(_pad_b64(salt_b64))
        expected = base64.urlsafe_b64decode(_pad_b64(hash_b64))
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(dk, expected) # hmac Secure Comparison instead of == to prevent timing attacks.
    except Exception:
        return False


def _pad_b64(s: str) -> str: # Restores required padding for base64 decoding.
    return s + "=" * (-len(s) % 4)


def new_token() -> str: # Creates secure session token.
    return secrets.token_urlsafe(32)


@dataclass(frozen=True) # Immutable data class representing authenticated user info. frozen=True Makes object immutable.
class AuthPrincipal:
    recruiter_id: int
    company: str
    username: str
