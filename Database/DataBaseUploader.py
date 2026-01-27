import os

import duckdb

# Robust path finding
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes DB is in the PARENT folder based on your output logs
db_path = os.path.abspath(os.path.join(current_dir, "..", "rag_staging.db"))


def inspect_data():
    print(f"Looking for database at: {db_path}")
    if not os.path.exists(db_path):
        print("❌ Database not found!")
        return

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    con.execute(f"ATTACH '{db_path}' AS rag (TYPE SQLITE);")

    print("\n--- Table Counts ---")
    con.sql("""
        SELECT 'Files' as table, count(*) as count FROM rag.files
        UNION ALL
        SELECT 'Chunks', count(*) FROM rag.chunks
        UNION ALL
        SELECT 'Enrichments', count(*) FROM rag.chunk_enrichments
    """).show()

    print("\n--- First 3 Chunks ---")
    con.sql("""
        SELECT chunk_id, substr(content, 1, 40) as preview, should_use
        FROM rag.chunks LIMIT 3
    """).show()

    print("\n--- Enrichment Progress ---")
    # New logic: Check if chunk exists in the enrichment table
    con.sql("""
        SELECT
            CASE WHEN e.chunk_id IS NOT NULL THEN 'Enriched' ELSE 'Pending' END as status,
            count(*) as count
        FROM rag.chunks c
        LEFT JOIN rag.chunk_enrichments e ON c.chunk_id = e.chunk_id
        WHERE c.should_use = 1
        GROUP BY 1
    """).show()


if __name__ == "__main__":
    inspect_data()
