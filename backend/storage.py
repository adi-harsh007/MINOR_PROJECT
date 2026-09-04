"""Upload file storage: naming, resolution, migration and cleanup.

Two problems this centralises:

* Records used to store an **absolute** host path. Moving the checkout,
  renaming a parent directory or putting the app in a container orphaned every
  historical image, because the stored string no longer pointed at anything.
  Paths are now stored relative to ``UPLOAD_DIR``.
* The uploaded file was written **before** inference and deleted afterwards on
  failure. A process killed in between left the file behind permanently, and
  nothing ever reclaimed it.
"""
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from .config import UPLOAD_DIR
from .logging_setup import get_logger

log = get_logger("storage")

# Files being written carry this suffix until the record that owns them is
# committed, so a sweep can tell an in-flight upload from an abandoned one.
PARTIAL_SUFFIX = ".part"

# An unreferenced file younger than this is assumed to belong to a request still
# in flight and is left alone.
ORPHAN_GRACE_SECONDS = 3600


# Exactly what new_upload_name() produces, and nothing else. The sweep deletes
# files, so it must recognise its own output rather than assume it owns every
# entry in the directory - an earlier version swept away the .gitkeep that keeps
# data/uploads/ in the repository.
_UPLOAD_NAME = re.compile(
    r"^[0-9a-f]{32}\.[A-Za-z0-9]{1,5}(?:" + re.escape(".part") + r")?$")


def new_upload_name(ext):
    """A fresh, collision-free storage name. The client filename is never used."""
    return "{}.{}".format(uuid.uuid4().hex, ext)


def is_upload_name(name):
    """Was this file created by us? Anything else is not ours to delete."""
    return bool(_UPLOAD_NAME.match(name))


def to_stored_path(absolute_path):
    """The value to persist: relative to UPLOAD_DIR when it lives there."""
    if not is_within_uploads(absolute_path):
        return absolute_path
    try:
        relative = os.path.relpath(os.path.abspath(absolute_path), UPLOAD_DIR)
    except ValueError:
        # Different drive on Windows - not under UPLOAD_DIR at all.
        return absolute_path
    if relative.startswith(os.pardir):
        return absolute_path
    return relative.replace(os.sep, "/")


def is_within_uploads(path):
    """Is `path` inside UPLOAD_DIR?

    commonpath raises ValueError when the two sit on different Windows drives -
    which is exactly the "the install moved" case this module exists to survive,
    so it must answer False rather than propagate.
    """
    root = os.path.abspath(UPLOAD_DIR)
    candidate = os.path.abspath(path)
    try:
        return os.path.commonpath([candidate, root]) == root
    except ValueError:
        return False


def resolve_stored_path(stored):
    """Absolute path for a persisted value.

    Accepts both the current relative form and the legacy absolute form, so
    records written before the migration keep resolving either way.

    Returns None for anything that would escape UPLOAD_DIR. The stored names are
    server-generated UUIDs, but this function feeds os.remove and must not be
    talked out of the directory it owns.
    """
    if not stored:
        return None

    candidate = stored if os.path.isabs(stored) else os.path.join(UPLOAD_DIR, stored)
    candidate = os.path.abspath(candidate)

    if not is_within_uploads(candidate):
        log.warning("stored path is outside the upload directory, ignoring: %s",
                    stored)
        return None
    return candidate


def delete_stored_file(stored):
    """Remove the file behind a record. Missing is success, not an error."""
    path = resolve_stored_path(stored)
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        log.warning("could not delete %s: %s", path, e)
        return False


def migrate_absolute_paths(connection):
    """Rewrite legacy absolute image_path values to the relative form.

    Runs at startup. Only rewrites paths that actually sit under UPLOAD_DIR;
    anything else is left untouched and reported, since a path pointing
    somewhere unexpected is a situation for a human, not a rewrite.
    """
    from sqlalchemy import text

    rows = connection.execute(
        text("SELECT id, image_path FROM diagnostic_sessions "
             "WHERE image_path IS NOT NULL")).fetchall()

    migrated = 0
    repaired = 0
    skipped = 0
    for row_id, stored in rows:
        if not os.path.isabs(stored):
            continue

        if is_within_uploads(stored):
            # Same install: express the path relative to the upload directory.
            relative = to_stored_path(stored)
        else:
            # The path points somewhere else entirely, which is what a moved or
            # containerised install looks like. If the file is present here under
            # the same name, adopt it - that is the case this migration exists
            # for. Otherwise leave the row alone; a path pointing at real data
            # elsewhere is a situation for a human, not a rewrite.
            name = os.path.basename(stored.replace("\\", "/"))
            if name and os.path.exists(os.path.join(UPLOAD_DIR, name)):
                relative = name
                repaired += 1
            else:
                skipped += 1
                continue

        connection.execute(
            text("UPDATE diagnostic_sessions SET image_path = :p WHERE id = :i"),
            {"p": relative, "i": row_id})
        migrated += 1

    if migrated:
        log.info("migrated %d image path(s) to storage-relative form "
                 "(%d re-homed from a previous install location)",
                 migrated, repaired)
    if skipped:
        log.warning("%d image path(s) point outside %s and no file of that name "
                    "is present here; left as-is for inspection",
                    skipped, UPLOAD_DIR)
    return migrated, skipped


def enforce_retention(connection, retention_days):
    """Delete sessions and images older than the retention window.

    Off unless configured (0 keeps everything), because silently deleting a
    clinician's records is worse than keeping them. When it does run it removes
    the file first and the row second: a row without its image degrades the
    history view, an image without its row is invisible and never reclaimed.
    """
    from sqlalchemy import text

    if not retention_days:
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    rows = connection.execute(
        text("SELECT id, image_path FROM diagnostic_sessions "
             "WHERE created_at IS NOT NULL AND created_at < :cutoff"),
        {"cutoff": cutoff.replace(tzinfo=None)}).fetchall()

    for row_id, stored in rows:
        delete_stored_file(stored)
        connection.execute(
            text("DELETE FROM diagnostic_sessions WHERE id = :i"), {"i": row_id})

    if rows:
        log.info("retention: removed %d session(s) older than %d day(s)",
                 len(rows), retention_days)
    return len(rows)


def sweep_orphans(connection, grace_seconds=ORPHAN_GRACE_SECONDS):
    """Delete upload files no record refers to, plus abandoned partial writes.

    Deliberately conservative: it only removes files that no row references
    *and* that are older than the grace period, so an upload belonging to a
    request still in flight is never taken out from under it.
    """
    from sqlalchemy import text

    if not os.path.isdir(UPLOAD_DIR):
        return 0

    # Match on the stored *name*, not on a resolved absolute path.
    #
    # Resolution can legitimately fail - a moved install, a legacy path on
    # another drive - and an earlier version treated every unresolvable row as
    # contributing no reference, so the sweep deleted files that records still
    # pointed at. Uploads are a flat directory of server-generated UUID names,
    # so the basename is an exact identity and cannot fail to compute.
    referenced = set()
    for (stored,) in connection.execute(
            text("SELECT image_path FROM diagnostic_sessions "
                 "WHERE image_path IS NOT NULL")).fetchall():
        referenced.add(os.path.normcase(os.path.basename(stored.replace("\\", "/"))))

    cutoff = time.time() - grace_seconds
    removed = 0
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.abspath(os.path.join(UPLOAD_DIR, name))
        if not os.path.isfile(path):
            continue
        if not is_upload_name(name):
            continue          # not ours: .gitkeep, a README, an operator's file
        if os.path.normcase(name) in referenced:
            continue
        try:
            if os.path.getmtime(path) > cutoff:
                continue          # young enough to still be in flight
            os.remove(path)
            removed += 1
        except OSError as e:
            log.warning("could not sweep %s: %s", path, e)

    if removed:
        log.info("swept %d orphaned upload file(s)", removed)
    return removed
