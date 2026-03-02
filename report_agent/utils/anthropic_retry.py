"""
Rate-limit-aware retry for Anthropic API calls.

The Anthropic SDK's built-in retry uses tiny exponential backoff (0.4s, 0.8s)
which is useless when the API returns retry-after: 60-120s.  This module
provides a wrapper that reads the ``retry-after`` header and sleeps for the
correct duration before retrying.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)

_DEFAULT_MAX_RETRIES = 3
_BUFFER_SECONDS = 5
_FALLBACK_WAIT = 65


def call_with_rate_limit_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    label: str = "Anthropic API",
) -> T:
    """
    Call *fn()* and retry on ``anthropic.RateLimitError``, respecting the
    ``retry-after`` response header.

    Args:
        fn: Zero-arg callable that makes the Anthropic API request.
        max_retries: How many times to retry after the initial attempt.
        label: Human-readable label for log messages.

    Returns:
        Whatever *fn()* returns on success.

    Raises:
        The original ``RateLimitError`` if all retries are exhausted,
        or any other exception immediately.
    """
    import anthropic

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except anthropic.RateLimitError as exc:
            if attempt >= max_retries:
                raise

            wait = _parse_retry_after(exc) + _BUFFER_SECONDS
            log.info(
                "%s rate-limited — waiting %ds before retry %d/%d",
                label, wait, attempt + 1, max_retries,
            )
            time.sleep(wait)


def _parse_retry_after(exc) -> int:
    """Extract the ``retry-after`` value (seconds) from the error response."""
    try:
        raw = exc.response.headers.get("retry-after", "")
        if raw:
            return int(float(raw))
    except Exception:
        pass
    return _FALLBACK_WAIT
