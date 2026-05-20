import sqlite3
from app.core.config import settings

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database with row_factory and foreign keys enabled."""
    con = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# Secondary indexes for lookups beyond PRIMARY KEY / UNIQUE constraints.
_RECRUITER_INDEXES = (
    (
        "idx_recruiters_username",
        "CREATE INDEX IF NOT EXISTS idx_recruiters_username ON recruiters(username)",
    ),
    (
        "idx_recruiters_created_at",
        "CREATE INDEX IF NOT EXISTS idx_recruiters_created_at ON recruiters(created_at)",
    ),
    (
        "idx_recruiter_sessions_recruiter_id",
        "CREATE INDEX IF NOT EXISTS idx_recruiter_sessions_recruiter_id "
        "ON recruiter_sessions(recruiter_id)",
    ),
    (
        "idx_recruiter_sessions_expires_at",
        "CREATE INDEX IF NOT EXISTS idx_recruiter_sessions_expires_at "
        "ON recruiter_sessions(expires_at)",
    ),
)


def _ensure_indexes(con: sqlite3.Connection) -> None:
    for _name, ddl in _RECRUITER_INDEXES:
        con.execute(ddl)


def ensure_db() -> None:
    """Initializes the database schema and indexes if they do not exist."""
    con = get_db_connection()
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS recruiters (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              company TEXT NOT NULL,
              username TEXT NOT NULL,
              password_hash TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              UNIQUE(company, username)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS recruiter_sessions (
              token TEXT PRIMARY KEY,
              recruiter_id INTEGER NOT NULL,
              expires_at INTEGER NOT NULL,
              created_at INTEGER NOT NULL,
              FOREIGN KEY (recruiter_id) REFERENCES recruiters(id) ON DELETE CASCADE
            )
            """
        )
        _ensure_indexes(con)
        con.commit()
    finally:
        con.close()
