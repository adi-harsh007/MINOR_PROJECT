"""In-process counters and latency samples.

Deliberately small: a bounded ring of recent durations plus a handful of
counters, held in memory and reset when the process restarts. Enough to answer
"how slow is inference right now, and what is the gate rejecting?" without
attaching an external metrics system to a research prototype.

Not a substitute for one: nothing here survives a restart or aggregates across
workers.
"""
import threading
import time
from collections import Counter, deque

# Enough samples to make a p95 meaningful without growing without bound.
_MAX_SAMPLES = 512

_lock = threading.Lock()
_counters = Counter()
_inference_ms = deque(maxlen=_MAX_SAMPLES)
_queue_ms = deque(maxlen=_MAX_SAMPLES)
_started_at = time.time()


def incr(name, amount=1):
    with _lock:
        _counters[name] += amount


def observe_inference_ms(value):
    with _lock:
        _inference_ms.append(float(value))


def observe_queue_ms(value):
    """Time spent waiting for an inference slot, not computing."""
    with _lock:
        _queue_ms.append(float(value))


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    index = int(len(sorted_values) * fraction)
    return sorted_values[min(index, len(sorted_values) - 1)]


def snapshot():
    """Current counters and latency distribution."""
    with _lock:
        counters = dict(_counters)
        samples = sorted(_inference_ms)
        queue_samples = sorted(_queue_ms)

    return {
        "uptime_seconds": round(time.time() - _started_at, 1),
        "counters": counters,
        "inference_ms": _distribution(samples),
        "queue_wait_ms": _distribution(queue_samples),
    }


def _distribution(samples):
    return {
        "count": len(samples),
        "p50": round(_percentile(samples, 0.50), 1) if samples else None,
        "p95": round(_percentile(samples, 0.95), 1) if samples else None,
        "max": round(samples[-1], 1) if samples else None,
    }


def reset():
    """Test helper; not exposed over HTTP."""
    with _lock:
        _counters.clear()
        _inference_ms.clear()
        _queue_ms.clear()
