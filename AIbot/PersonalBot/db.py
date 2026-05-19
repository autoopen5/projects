import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent / "personal.db"


def init():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL UNIQUE,
                weight      REAL,
                activities  TEXT DEFAULT '',
                pages_read  INTEGER DEFAULT 0,
                notes       TEXT DEFAULT ''
            )
        """)


def _upsert(day: str, **fields):
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute("SELECT * FROM logs WHERE date = ?", (day,)).fetchone()
        if row:
            sets = ", ".join(f"{k} = ?" for k in fields)
            c.execute(f"UPDATE logs SET {sets} WHERE date = ?", (*fields.values(), day))
        else:
            cols = "date, " + ", ".join(fields)
            vals = "?, " + ", ".join("?" * len(fields))
            c.execute(f"INSERT INTO logs ({cols}) VALUES ({vals})", (day, *fields.values()))


def log_weight(weight: float, day: str = None):
    _upsert(day or date.today().isoformat(), weight=weight)


def log_activity(activity: str, day: str = None):
    day = day or date.today().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute("SELECT activities FROM logs WHERE date = ?", (day,)).fetchone()
        existing = row[0] if row else ""
        merged = (existing + ", " + activity).strip(", ") if existing else activity
    _upsert(day, activities=merged)


def log_pages(pages: int, day: str = None):
    day = day or date.today().isoformat()
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute("SELECT pages_read FROM logs WHERE date = ?", (day,)).fetchone()
        total = (row[0] or 0) + pages if row else pages
    _upsert(day, pages_read=total)


def get_last_weight() -> float | None:
    with sqlite3.connect(DB_PATH) as c:
        row = c.execute(
            "SELECT weight FROM logs WHERE weight IS NOT NULL ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None


def get_recent_logs(days: int = 7) -> list[dict]:
    with sqlite3.connect(DB_PATH) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(f"""
            SELECT date, weight, activities, pages_read
            FROM logs
            WHERE date >= date('now', '-{days} days')
            ORDER BY date DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_week_summary() -> list[dict]:
    return get_recent_logs(7)
