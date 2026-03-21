"""Rate limiting configuration for the API.

Uses Redis-backed storage when available for cross-instance coordination.
Falls back to in-memory storage for local development.
"""

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

if settings.redis_enabled and settings.redis_url:
    logger.info("Rate limiter using Redis: %s", settings.redis_url)
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis_url,
    )
else:
    logger.info("Rate limiter using in-memory storage (single-instance only)")
    limiter = Limiter(key_func=get_remote_address)
