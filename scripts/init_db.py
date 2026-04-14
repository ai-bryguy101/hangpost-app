#!/usr/bin/env python3
"""Initialize the Hangpost database with the schema.

Creates the SQLite database file and all tables/indexes. Safe to run
multiple times — all CREATE statements use IF NOT EXISTS.

Usage:
    python scripts/init_db.py                          # default: data/hangpost.db
    python scripts/init_db.py --db data/custom.db      # custom path

WHAT THIS DOES:
1. Creates the data/ directory if it doesn't exist
2. Creates (or opens) the SQLite database file
3. Runs the schema SQL to create tables and indexes
4. Prints a summary of what was created

WHY A SEPARATE SCRIPT:
In production, schema creation is a controlled migration step, not something
that happens automatically on every app startup. Having it as a standalone
script makes it explicit: you run it once when setting up a new environment,
and the schema is ready. The migrate_csv_to_db.py script calls init_schema()
too, but this script is useful when you want to create an empty database
without importing any data.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hangpost_matching.db import get_connection, init_schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the Hangpost database schema.")
    parser.add_argument(
        "--db", default="data/hangpost.db",
        help="Path to SQLite database file (default: data/hangpost.db)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)

    # Create parent directory if needed.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    existed_before = db_path.exists()
    conn = get_connection(db_path)
    init_schema(conn)
    conn.close()

    if existed_before:
        print(f"Schema updated (tables created if missing) -> {db_path}")
    else:
        print(f"New database created with schema -> {db_path}")

    # Print table list for verification.
    conn = get_connection(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print(f"Tables: {', '.join(row['name'] for row in tables)}")

    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%' ORDER BY name"
    ).fetchall()
    print(f"Indexes: {', '.join(row['name'] for row in indexes)}")
    conn.close()


if __name__ == "__main__":
    main()
