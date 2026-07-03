"""
Async Redis client for InSightATS result caching.

Design notes
------------
- Uses redis.asyncio (bundled in redis>=4.2 / redis[hiredis]>=5.x).
- hiredis C-extension is selected automatically when installed — gives ~5–10×
  faster response parsing vs the pure-Python parser.
- The client is created lazily on first use and stored in a module-level
  variable so it is shared across all FastAPI requests (connection pooling).
- If REDIS_URL is empty or Redis is unreachable, every public function falls
  back silently — the app runs without caching. Never raises to a caller.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — shared across all async workers in a single process.
_redis: Optional["redis.asyncio.Redis"] = None  # type: ignore[name-defined]


async def get_redis() -> Optional["redis.asyncio.Redis"]:  # type: ignore[name-defined]
    """
    Return the shared Redis client, creating it on first call.

    Returns ``None`` if REDIS_URL is not configured or the connection fails.
    """
    global _redis

    if not settings.REDIS_URL:
        return None

    if _redis is None:
        try:
            # pyrefly: ignore [missing-import]
            import redis.asyncio as aioredis  # noqa: PLC0415

            _redis = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,   # fail fast on misconfigured URL
                socket_timeout=2,
                retry_on_timeout=False,
            )
            # Ping once to verify connectivity at startup.
            await _redis.ping()
            logger.info("Redis connected: %s", settings.REDIS_URL)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable — caching disabled: %s", exc)
            _redis = None

    return _redis


async def cache_get(key: str) -> Optional[str]:
    """
    Fetch a cached string value.

    Returns ``None`` on cache miss or when Redis is unavailable.
    """
    client = await get_redis()
    if client is None:
        return None
    try:
        return await client.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis GET error (key=%s): %s", key, exc)
        return None


async def cache_set(key: str, value: str, ttl: int = 3600) -> None:
    """
    Store *value* under *key* with the given TTL (seconds).

    No-ops silently when Redis is unavailable.
    """
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(key, value, ex=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Redis SET error (key=%s): %s", key, exc)


async def close_redis() -> None:
    """
    Close the Redis connection pool.

    Call this from the FastAPI lifespan shutdown handler.
    """
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:  # noqa: BLE001
            pass
        _redis = None
        logger.info("Redis connection closed.")
