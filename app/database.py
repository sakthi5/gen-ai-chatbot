from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./chatbot.db"


# The engine is responsible for connecting our Python application to the database.


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)


@event.listens_for(engine, "connect")
def _configure_sqlite_connection(dbapi_connection, connection_record):
    """Avoid "database is locked" errors under any concurrent access.

    SQLite's defaults are unforgiving here: if one connection is writing
    when another tries to write (or even read, under the default journal
    mode), the second one fails immediately with OperationalError instead
    of just waiting — this app hit exactly that on both /chat (saving the
    reply) and DELETE /conversations/{id}, even from ordinary use (a
    dev-tool script and the live app both touching the same chatbot.db,
    or simply two requests landing close together).

    - WAL (Write-Ahead Logging) journal mode lets readers and writers work
      concurrently instead of blocking each other.
    - busy_timeout tells SQLite to retry for up to 5s before giving up,
      instead of failing the instant it meets any contention at all.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()