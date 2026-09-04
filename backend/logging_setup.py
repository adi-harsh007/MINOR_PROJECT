"""Application logging.

Minimal on purpose: enough to make a failure traceable from the reference a user
was shown back to a full traceback in the server log. Structured/JSON output and
a per-request id on *every* request are a separate piece of work.

The backend otherwise uses bare `print()`, which has no levels, no timestamps and
no way to tell an operational error from a status line.
"""
import contextvars
import logging
import os
import re
import sys
import uuid

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_ROOT_NAME = "dermascan"
_configured = False

# The id of the request being served on this task. A context variable rather
# than thread-local state: analyses run in a threadpool but the middleware that
# sets this runs on the event loop, and contextvars propagate across both.
_request_id = contextvars.ContextVar("request_id", default="-")

# A client may supply its own id to correlate with its logs, but it is untrusted
# input that ends up in every log line for the request, so anything outside this
# shape is replaced rather than echoed.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id():
    """A fresh id. Twelve hex characters: short enough to quote over the phone."""
    return uuid.uuid4().hex[:12]


def adopt_request_id(candidate):
    """Use the caller's id when it is safe to, otherwise mint one."""
    if candidate and _SAFE_REQUEST_ID.match(candidate):
        return candidate
    return new_request_id()


def set_request_id(value):
    return _request_id.set(value)


def reset_request_id(token):
    _request_id.reset(token)


def get_request_id():
    return _request_id.get()


def _install_record_factory():
    """Stamp every LogRecord with the current request id.

    A record factory rather than a Filter: filters attached to a logger are not
    applied to records propagating up from child loggers, and filters attached
    to a handler are invisible to any other handler - including pytest's caplog
    and uvicorn's own. The factory runs wherever a record is created, so the id
    is always present and any formatter or consumer can use it.
    """
    existing = logging.getLogRecordFactory()
    if getattr(existing, "_dermascan_request_id", False):
        return

    def factory(*args, **kwargs):
        record = existing(*args, **kwargs)
        record.request_id = _request_id.get()
        return record

    factory._dermascan_request_id = True
    logging.setLogRecordFactory(factory)


def configure_logging():
    """Attach a handler to the application logger exactly once.

    Uvicorn configures its own loggers but leaves the root logger bare, so
    without this the application's records would be silently dropped depending
    on how the process was started.
    """
    global _configured
    if _configured:
        return
    _install_record_factory()
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(LOG_LEVEL)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
        logger.addHandler(handler)
    # Uvicorn's handlers would otherwise print every record a second time.
    logger.propagate = False
    _configured = True


def get_logger(name):
    """Logger for a submodule, e.g. get_logger("diagnostics")."""
    configure_logging()
    return logging.getLogger("{}.{}".format(_ROOT_NAME, name))
