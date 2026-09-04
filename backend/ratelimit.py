"""Per-client request limiting.

A fixed-window counter per client address, held in memory. Enough to stop one
caller monopolising a single-process CPU-bound service; not a substitute for a
real edge rate limiter, and it does not aggregate across workers.

Inference is the expensive path and gets its own, tighter budget: a caller can
poll the history view freely without being allowed to queue hundreds of scans.
"""
import threading
import time

from .config import RATE_LIMIT_PER_MINUTE, ANALYZE_RATE_LIMIT_PER_MINUTE
from .logging_setup import get_logger

log = get_logger("ratelimit")

_WINDOW_SECONDS = 60.0

_lock = threading.Lock()
# client -> {bucket_name: (window_start, count)}
_hits = {}
# Bound the table so a spray of forged addresses cannot grow it without limit.
_MAX_TRACKED_CLIENTS = 4096


def _limit_for(path):
    if path.startswith("/api/analyze"):
        return "analyze", ANALYZE_RATE_LIMIT_PER_MINUTE
    return "api", RATE_LIMIT_PER_MINUTE


def check(client, path, now=None):
    """Returns (allowed, retry_after_seconds, limit).

    A limit of 0 disables that bucket.
    """
    bucket, limit = _limit_for(path)
    if not limit:
        return True, 0, 0

    now = now if now is not None else time.monotonic()
    with _lock:
        if len(_hits) > _MAX_TRACKED_CLIENTS:
            _evict_stale(now)

        buckets = _hits.setdefault(client, {})
        window_start, count = buckets.get(bucket, (now, 0))

        if now - window_start >= _WINDOW_SECONDS:
            window_start, count = now, 0

        if count >= limit:
            retry_after = max(1, int(_WINDOW_SECONDS - (now - window_start)) + 1)
            buckets[bucket] = (window_start, count)
            return False, retry_after, limit

        buckets[bucket] = (window_start, count + 1)
        return True, 0, limit


def _evict_stale(now):
    """Drop clients whose windows have all expired. Caller holds the lock."""
    stale = [
        client for client, buckets in _hits.items()
        if all(now - start >= _WINDOW_SECONDS for start, _ in buckets.values())
    ]
    for client in stale:
        del _hits[client]
    if not stale:
        # Everything is live; clear the oldest half rather than grow unbounded.
        for client in list(_hits)[: len(_hits) // 2]:
            del _hits[client]


def reset():
    """Test helper."""
    with _lock:
        _hits.clear()
