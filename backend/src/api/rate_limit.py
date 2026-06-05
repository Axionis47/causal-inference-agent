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

# Local development polls the API freely (live job views, token counter,
# dataset panel). Throttling that only produces noise, so the limiter is
# disabled outside staging/production. Real limits still apply when deployed.
_rate_limit_enabled = settings.environment != "development"

if settings.redis_enabled and settings.redis_url:
    logger.info("Rate limiter using Redis: %s", settings.redis_url)
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=settings.redis_url,
        enabled=_rate_limit_enabled,
    )
else:
    logger.info(
        "Rate limiter using in-memory storage (single-instance only), enabled=%s",
        _rate_limit_enabled,
    )
    limiter = Limiter(key_func=get_remote_address, enabled=_rate_limit_enabled)
