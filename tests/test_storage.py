"""Upload storage: path form, migration, and the orphan sweep.

The sweep deletes files. An earlier version of it built its set of referenced
files from resolved absolute paths, so any row whose path failed to resolve -
a moved install, a legacy path on another drive - contributed no reference and
its image was deleted as an orphan. These tests exist mostly to keep that from
coming back.
"""
import os
import time

import pytest
from sqlalchemy import text

import backend.storage as storage
from backend.database import engine


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Point the storage module at a throwaway upload directory."""
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(directory))
    return directory


# Realistic storage names. The sweep only recognises its own naming (32 hex
# characters plus an extension), so a test using an arbitrary filename would be
# spared by that filter and pass without exercising anything.
ORPHAN = "a" * 32 + ".jpg"
KEPT = "c" * 32 + ".jpg"
UNRESOLVABLE = "d" * 32 + ".jpg"
INFLIGHT = "e" * 32 + ".jpg"
LEGACY_ABSOLUTE = "Z:" + chr(92) + "gone" + chr(92) + "uploads" + chr(92) + UNRESOLVABLE


def make_file(directory, name, age_seconds=0):
    path = directory / name
    path.write_bytes(b"jpegdata")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


def rows(connection):
    return dict(connection.execute(
        text("SELECT id, image_path FROM diagnostic_sessions")).fetchall())


def insert(connection, row_id, image_path):
    connection.execute(
        text("INSERT INTO diagnostic_sessions (id, image_path, status) "
             "VALUES (:i, :p, 'completed')"),
        {"i": row_id, "p": image_path})


# ── path handling ────────────────────────────────────────────────────────

def test_paths_are_stored_relative_to_the_upload_directory(uploads):
    absolute = os.path.join(str(uploads), "abc123.jpg")
    assert storage.to_stored_path(absolute) == "abc123.jpg"


def test_relative_paths_resolve_back_to_the_current_upload_directory(uploads):
    resolved = storage.resolve_stored_path("abc123.jpg")
    assert resolved == os.path.join(str(uploads), "abc123.jpg")


def test_paths_outside_the_upload_directory_are_refused(uploads):
    """resolve_stored_path feeds os.remove and must not be led out of its tree."""
    assert storage.resolve_stored_path("../../etc/passwd") is None
    assert storage.resolve_stored_path(os.path.join("..", "secret.jpg")) is None


def test_resolution_survives_a_path_on_another_drive(uploads):
    """A legacy path from a different install must answer None, not raise.

    os.path.commonpath raises ValueError across Windows drives, which is exactly
    the moved-install case this module exists to handle.
    """
    assert storage.resolve_stored_path(r"Z:\elsewhere\uploads\abc.jpg") is None


# ── migration ────────────────────────────────────────────────────────────

def test_absolute_paths_from_this_install_become_relative(uploads):
    make_file(uploads, "same.jpg")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9001, os.path.join(str(uploads), "same.jpg"))
        storage.migrate_absolute_paths(conn)
        assert rows(conn)[9001] == "same.jpg"
        conn.execute(text("DELETE FROM diagnostic_sessions"))


def test_paths_from_a_previous_install_are_rehomed_by_name(uploads):
    """The image moved with the install; adopt it under the same name."""
    make_file(uploads, "moved.jpg")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9002, r"D:\an\old\install\data\uploads\moved.jpg")
        storage.migrate_absolute_paths(conn)
        assert rows(conn)[9002] == "moved.jpg"
        conn.execute(text("DELETE FROM diagnostic_sessions"))


def test_paths_with_no_matching_file_are_left_alone(uploads):
    """Pointing at real data elsewhere is a job for a human, not a rewrite."""
    original = r"D:\somewhere\else\absent.jpg"
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9003, original)
        storage.migrate_absolute_paths(conn)
        assert rows(conn)[9003] == original
        conn.execute(text("DELETE FROM diagnostic_sessions"))


# ── the orphan sweep ─────────────────────────────────────────────────────

def test_sweep_removes_genuinely_unreferenced_files(uploads):
    make_file(uploads, ORPHAN, age_seconds=7200)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == 1
    assert not (uploads / ORPHAN).exists()


def test_sweep_spares_files_a_record_still_refers_to(uploads):
    make_file(uploads, KEPT, age_seconds=7200)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9004, KEPT)
        assert storage.sweep_orphans(conn) == 0
        conn.execute(text("DELETE FROM diagnostic_sessions"))
    assert (uploads / KEPT).exists()


def test_sweep_spares_referenced_files_whose_paths_cannot_resolve(uploads):
    """The regression that deleted real images.

    The row's stored path is an absolute one from another install, so resolving
    it against this UPLOAD_DIR yields nothing. The file is still referenced and
    must survive.
    """
    make_file(uploads, UNRESOLVABLE, age_seconds=7200)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9005, LEGACY_ABSOLUTE)
        assert storage.resolve_stored_path(LEGACY_ABSOLUTE) is None
        assert storage.sweep_orphans(conn) == 0
        conn.execute(text("DELETE FROM diagnostic_sessions"))
    assert (uploads / UNRESOLVABLE).exists()


def test_sweep_spares_files_young_enough_to_be_in_flight(uploads):
    make_file(uploads, INFLIGHT, age_seconds=0)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == 0
    assert (uploads / INFLIGHT).exists()


def test_sweep_only_touches_files_it_created(uploads):
    """The sweep once deleted data/uploads/.gitkeep, a tracked repository file.

    It must recognise its own naming (32 hex characters plus an extension) and
    leave everything else in the directory alone, however old and unreferenced.
    """
    keep = uploads / ".gitkeep"
    keep.write_bytes(b"")
    readme = make_file(uploads, "README.md", age_seconds=7200)
    operator_file = make_file(uploads, "notes.txt", age_seconds=7200)
    old = time.time() - 7200
    os.utime(keep, (old, old))

    real_orphan = make_file(uploads, ORPHAN, age_seconds=7200)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        removed = storage.sweep_orphans(conn)

    assert removed == 1
    assert not real_orphan.exists()
    assert keep.exists()
    assert readme.exists()
    assert operator_file.exists()


def test_abandoned_partial_writes_are_reclaimed(uploads):
    """A process killed mid-upload leaves a .part file with no owner."""
    partial = make_file(uploads, INFLIGHT + storage.PARTIAL_SUFFIX,
                        age_seconds=7200)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == 1
    assert not partial.exists()


# ── retention ────────────────────────────────────────────────────────────

def insert_aged(connection, row_id, image_path, days_old):
    from datetime import datetime, timedelta, timezone
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    connection.execute(
        text("INSERT INTO diagnostic_sessions (id, image_path, status, created_at) "
             "VALUES (:i, :p, 'completed', :c)"),
        {"i": row_id, "p": image_path, "c": created.replace(tzinfo=None)})


def test_retention_is_off_unless_configured(uploads):
    """Silently deleting a clinician's records is worse than keeping them."""
    make_file(uploads, ORPHAN)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert_aged(conn, 9010, ORPHAN, days_old=400)
        assert storage.enforce_retention(conn, 0) == 0
        assert 9010 in rows(conn)
        conn.execute(text("DELETE FROM diagnostic_sessions"))
    assert (uploads / ORPHAN).exists()


def test_retention_removes_old_sessions_and_their_images(uploads):
    make_file(uploads, ORPHAN)
    make_file(uploads, KEPT)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert_aged(conn, 9011, ORPHAN, days_old=45)
        insert_aged(conn, 9012, KEPT, days_old=5)

        assert storage.enforce_retention(conn, 30) == 1

        remaining = rows(conn)
        assert 9011 not in remaining
        assert 9012 in remaining
        conn.execute(text("DELETE FROM diagnostic_sessions"))

    assert not (uploads / ORPHAN).exists()
    assert (uploads / KEPT).exists()


# ── cross-platform paths ─────────────────────────────────────────────────
# These assertions must hold identically on Windows and on Linux. os.path.isabs
# only recognises the host's own convention, so a database written on one and
# served on the other had every stored path misread as a bare filename - the
# exact "install moved" case this module exists to survive. It also made the
# suite pass locally and fail in CI.

WINDOWS_PATH = "D:" + chr(92) + "app" + chr(92) + "data" + chr(92) + "uploads" \
    + chr(92) + ORPHAN
POSIX_PATH = "/srv/app/data/uploads/" + ORPHAN
UNC_PATH = chr(92) * 2 + "server" + chr(92) + "share" + chr(92) + ORPHAN


@pytest.mark.parametrize("path", [WINDOWS_PATH, POSIX_PATH, UNC_PATH])
def test_foreign_absolute_paths_are_recognised_on_any_host(path):
    assert storage.looks_absolute(path) is True


@pytest.mark.parametrize("path", [ORPHAN, "sub/dir/" + ORPHAN, ""])
def test_relative_paths_are_not_mistaken_for_absolute(path):
    assert storage.looks_absolute(path) is False


@pytest.mark.parametrize("path", [WINDOWS_PATH, POSIX_PATH, UNC_PATH])
def test_a_path_from_another_install_never_resolves_into_this_one(uploads, path):
    """It must answer None, not silently join itself under UPLOAD_DIR.

    Joined, it would name a file inside the upload directory that this record
    does not own - and resolve_stored_path feeds os.remove.
    """
    assert storage.resolve_stored_path(path) is None


@pytest.mark.parametrize("path", [WINDOWS_PATH, POSIX_PATH])
def test_migration_rehomes_a_path_from_another_platform(uploads, path):
    """A database carried between Windows and Linux still finds its images."""
    make_file(uploads, ORPHAN)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        insert(conn, 9020, path)
        storage.migrate_absolute_paths(conn)
        assert rows(conn)[9020] == ORPHAN
        conn.execute(text("DELETE FROM diagnostic_sessions"))


# ── refusing to sweep in bulk ────────────────────────────────────────────
# The sweep reclaims the occasional file a crashed request left behind. A sweep
# that wants to take most of the directory means the database being consulted
# does not describe this upload directory - DATABASE_URL pointed somewhere new,
# the database file lost or reset, a throwaway database opened against the real
# uploads. That last one is not hypothetical: it deleted 154 uploads whose real
# records still referenced them, at startup, with no error.

def _fill(directory, count, age_seconds=7200):
    names = []
    for i in range(count):
        name = "{:032x}.jpg".format(i)
        make_file(directory, name, age_seconds=age_seconds)
        names.append(name)
    return names


def test_sweep_refuses_when_the_database_references_nothing(uploads):
    """An empty database cannot authorise deleting a directory full of images."""
    names = _fill(uploads, 40)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == 0
    assert all((uploads / n).exists() for n in names)


def test_sweep_refuses_when_most_of_the_directory_is_unreferenced(uploads):
    """Five rows against fifty files is a mismatch, not fifty-five orphans."""
    names = _fill(uploads, 50)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        for i, name in enumerate(names[:5]):
            insert(conn, 9100 + i, name)
        assert storage.sweep_orphans(conn) == 0
        conn.execute(text("DELETE FROM diagnostic_sessions"))
    assert all((uploads / n).exists() for n in names)


def test_a_lone_stray_file_is_still_housekeeping(uploads):
    """Magnitude is the signal, not emptiness.

    A fresh install whose first upload crashed has no rows and one stray file,
    and reclaiming it is exactly what the sweep is for. Refusing on an empty
    database alone would leak those forever.
    """
    make_file(uploads, ORPHAN, age_seconds=7200)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == 1
    assert not (uploads / ORPHAN).exists()


def test_bulk_sweep_can_be_forced_once_the_operator_is_certain(uploads,
                                                              monkeypatch):
    """The refusal is a guard, not a wall - but it has to be asked for."""
    names = _fill(uploads, 40)
    monkeypatch.setattr(storage, "ALLOW_BULK_SWEEP", True)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        assert storage.sweep_orphans(conn) == len(names)
    assert not any((uploads / n).exists() for n in names)


def test_refusing_to_sweep_leaves_every_file_intact(uploads):
    """Nothing is half-deleted: the decision is made before any file is removed."""
    names = _fill(uploads, 40)
    before = {n: (uploads / n).read_bytes() for n in names}
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM diagnostic_sessions"))
        storage.sweep_orphans(conn)
    assert {n: (uploads / n).read_bytes() for n in names} == before
