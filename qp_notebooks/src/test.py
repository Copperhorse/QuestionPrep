import marimo

__generated_with = "0.22.4"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sqlalchemy

    DATABASE_URL = (
        "sqlite:////home/copper/Desktop/Project/QuestionPrep/data/rag_staging.db"
    )
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _(engine, files, mo):
    _df = mo.sql(
        f"""
        Select * FROm files
        """,
        engine=engine
    )
    return


@app.cell
def _(chunks, engine, mo):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            chunks
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_enrichments, engine, mo):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            chunk_enrichments
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_questions, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * 
        FROM chunk_questions 
        WHERE question_text LIKE '%space complexity of EXPANDER%O(1) per node%'
           OR question_text LIKE '%data collection relate to the challenges in machine learning applications%';
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


@app.cell
def _(chunk_enrichments, chunk_questions, chunks, engine, mo):
    _df = mo.sql(
        f"""
        SELECT 
            q.question_text, 
            q.answer_text, 
            q.source_quote,
            c.content,
            c.chunk_id,
            e.summary,
            c.file_id
        FROM chunk_questions q
        JOIN chunks c 
            ON q.chunk_id = c.chunk_id
        JOIN chunk_enrichments e 
            ON e.chunk_id = c.chunk_id
        WHERE c.should_use = 1 
        ORDER by e.processed_at DESC
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_rejections, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM chunk_rejections
        """,
        engine=engine
    )
    return


@app.cell(hide_code=True)
def _(chunk_enrichments, chunk_questions, chunks, engine, mo):
    _df = mo.sql(
        f"""
                    SELECT q.question_id, q.chunk_id, c.file_id, q.question_text, q.answer_text,
                           q.source_quote, q.difficulty, q.question_type, e.tags
                    FROM chunk_questions q
                    JOIN chunks c ON q.chunk_id = c.chunk_id
                    LEFT JOIN chunk_enrichments e ON q.chunk_id = e.chunk_id
                    WHERE c.file_id = 'e473bee9-5647-40ed-af26-9386c4ed4fb1'
                    ORDER BY c.chunk_index
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
