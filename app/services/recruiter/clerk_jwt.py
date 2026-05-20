"""Verify Clerk session JWTs via JWKS (RS256 signature + issuer + expiry)."""

from __future__ import annotations

import jwt
from jwt import PyJWKClient, PyJWTError

from app.core.config import settings

_jwks_client: PyJWKClient | None = None


def _looks_like_jwt(token: str) -> bool:
    parts = token.split(".")
    return len(parts) == 3 and all(parts)


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not settings.CLERK_JWKS_URL:
            raise RuntimeError(
                "Clerk JWT verification is not configured. "
                "Set CLERK_ISSUER (e.g. https://your-app.clerk.accounts.dev) "
                "or CLERK_JWKS_URL on the server."
            )
        _jwks_client = PyJWKClient(settings.CLERK_JWKS_URL, cache_keys=True)
    return _jwks_client


def verify_clerk_jwt(token: str) -> dict:
    """
    Validate Clerk-issued Bearer token:
    - RS256 signature against Clerk JWKS
    - issuer (CLERK_ISSUER)
    - exp / sub required
    - optional audience (CLERK_AUDIENCE) when set
    """
    if not settings.CLERK_ISSUER:
        raise ValueError("CLERK_ISSUER is not configured")

    try:
        client = _get_jwks_client()
        signing_key = client.get_signing_key_from_jwt(token)
        decode_options = {"require": ["exp", "sub"]}
        kwargs: dict = {
            "algorithms": ["RS256"],
            "issuer": settings.CLERK_ISSUER,
            "options": decode_options,
        }
        if settings.CLERK_AUDIENCE:
            kwargs["audience"] = settings.CLERK_AUDIENCE
            decode_options["verify_aud"] = True
        else:
            decode_options["verify_aud"] = False

        return jwt.decode(token, signing_key.key, **kwargs)
    except PyJWTError as e:
        raise ValueError(f"Invalid Clerk token: {e}") from e
