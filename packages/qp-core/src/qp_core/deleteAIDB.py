import sqlite3
from pathlib import Path

# Adjust this path to your actual DB location
DB_PATH = Path("/home/copper/Desktop/Project/QuestionPrep/data/rag_staging.db")


def nuke_enrichment_data():
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    tables_to_clear = ["chunk_questions", "chunk_enrichments", "chunk_rejections"]

    try:
        print("🧨 Initializing database wipe...")

        for table in tables_to_clear:
            # 1. Check if the table even exists before trying to delete
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            )
            if not cursor.fetchone():
                print(f"⏩ Skipping {table} (Table doesn't exist yet)")
                continue

            # 2. Delete all rows
            cursor.execute(f"DELETE FROM {table}")
            print(f"✅ Cleared: {table}")

            # 3. Reset the ID counter ONLY if sqlite_sequence exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
            )
            if cursor.fetchone():
                cursor.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
                print(f"🔄 Reset ID sequence for: {table}")

        conn.commit()
        print(
            "\n✨ Database is clean. The 'Chef' is gone. You are ready for a fresh run."
        )

    except Exception as e:
        print(f"❌ Error during nuke: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    confirm = input(
        "This will delete ALL AI-generated questions and summaries. Type 'yes' to proceed: "
    )
    if confirm.lower() == "yes":
        nuke_enrichment_data()
    else:
        print("Operation cancelled.")
