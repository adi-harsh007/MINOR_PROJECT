from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL
from .logging_setup import get_logger

log = get_logger("database")

_IS_SQLITE = DATABASE_URL.startswith("sqlite")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})


if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, connection_record):
        """Make SQLite survive concurrent access.

        Two settings, for two different failures:

        * ``journal_mode=WAL`` lets readers proceed while a writer holds the
          database. Under the default rollback journal a single write blocks
          every reader, so a scan being recorded stalls the history view.
        * ``busy_timeout`` makes a blocked writer wait instead of failing
          immediately. The default is 0 - a concurrent write raises
          "database is locked" straight away rather than retrying.

        Now that analyses run concurrently in a threadpool, both are reachable.
        WAL is persisted in the database file; the timeout is per connection,
        which is why this runs on every connect.
        """
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA busy_timeout=5000")
            # Unavailable for in-memory databases and on some network shares;
            # the returned mode says what was actually applied.
            cursor.execute("PRAGMA journal_mode=WAL")
            mode = cursor.fetchone()
            if mode and mode[0].lower() != "wal":
                log.warning("SQLite journal_mode is %s, not WAL; "
                            "concurrent readers will block on writes", mode[0])
            # Safe to relax under WAL: a crash can lose the last transaction but
            # cannot corrupt the database.
            cursor.execute("PRAGMA synchronous=NORMAL")
        finally:
            cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _add_missing_columns():
    """Add columns introduced after a database was first created.

    Base.metadata.create_all() only creates missing *tables*; it never alters an
    existing one, so a database made before a column existed would keep failing
    on every query that referenced it.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "diagnostic_sessions" not in inspector.get_table_names():
        return

    existing = {c["name"] for c in inspector.get_columns("diagnostic_sessions")}
    additions = {
        "anatomic_site": "VARCHAR(64)",
        "melanoma_alert": "BOOLEAN",
        "melanoma_probability": "FLOAT",
    }

    with engine.begin() as conn:
        for name, ddl_type in additions.items():
            if name not in existing:
                conn.execute(text(
                    f"ALTER TABLE diagnostic_sessions ADD COLUMN {name} {ddl_type}"))
                log.info("database migrated: added column %s", name)


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _migrate_and_sweep_storage()


def _migrate_and_sweep_storage():
    """Bring stored upload paths up to date and reclaim abandoned files.

    Both are safe to run on every start: the migration only touches rows still
    holding an absolute path, and the sweep only removes files that no row
    references and that are older than its grace period.
    """
    from sqlalchemy import inspect

    from .config import UPLOAD_RETENTION_DAYS
    from .storage import (migrate_absolute_paths, sweep_orphans,
                          enforce_retention)

    if "diagnostic_sessions" not in inspect(engine).get_table_names():
        return

    try:
        with engine.begin() as conn:
            migrate_absolute_paths(conn)
            enforce_retention(conn, UPLOAD_RETENTION_DAYS)
            sweep_orphans(conn)
    except Exception as e:
        # Storage housekeeping must never stop the API from coming up.
        log.warning("storage maintenance skipped: %s", e)
