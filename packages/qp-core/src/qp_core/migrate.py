import os
import sqlite3

import pandas as pd

DB_FILE = "rag_staging.db"
CHUNKS_CSV = "output/chunks.csv"
FILES_CSV = "output/files.csv"


def migrate_csv_to_db():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("Removed old database. Starting fresh.")

    from DBManager import DBManager

    db = DBManager(DB_FILE)  # Init new schema (2 tables)
    conn = sqlite3.connect(DB_FILE)

    # 1. Files
    if os.path.exists(FILES_CSV):
        print("Migrating files...")
        df = pd.read_csv(FILES_CSV)
        valid = [r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()]
        df = df[[c for c in df.columns if c in valid]]
        df.to_sql("files", conn, if_exists="append", index=False)

    # 2. Chunks (CLEAN)
    if os.path.exists(CHUNKS_CSV):
        print("Migrating chunks...")
        df = pd.read_csv(CHUNKS_CSV)

        # Normalize Boolean
        if "should_use" in df.columns:
            df["should_use"] = df["should_use"].astype(int)

        # Drop columns we don't want
        drop_cols = [
            "parent_section",
            "top_header",
            "tags",
            "summary",
            "questions",
        ]  # + any old enrichment cols
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

        # Filter to exact DB schema
        valid = [r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()]
        df = df[[c for c in df.columns if c in valid]]

        df.to_sql("chunks", conn, if_exists="append", index=False)
        print(f"Migrated {len(df)} chunks.")

    conn.close()
    print("Migration complete!")


if __name__ == "__main__":
    migrate_csv_to_db()
