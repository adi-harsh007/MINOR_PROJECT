from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
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
    additions = {"anatomic_site": "VARCHAR(64)"}

    with engine.begin() as conn:
        for name, ddl_type in additions.items():
            if name not in existing:
                conn.execute(text(
                    f"ALTER TABLE diagnostic_sessions ADD COLUMN {name} {ddl_type}"))
                print(f"Database migrated: added column {name}")


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
