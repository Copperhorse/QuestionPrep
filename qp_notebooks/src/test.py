import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full", auto_download=["ipynb"])


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sqlalchemy

    DATABASE_URL = "sqlite:////home/copper/Desktop/QuestionPrep/QuestionPrep/data/rag_staging.db"
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
        SELECT * FROM chunks
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_enrichments, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM chunk_enrichments
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_questions, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM chunk_questions
        """,
        engine=engine
    )
    return


@app.cell
def _(chunk_enrichments, chunk_questions, chunks, engine, mo):
    _df = mo.sql(
        f"""
        SELECT 
            e.processed_at,
            e.tags,
            c.file_id,
            c.content,
            c.chunk_index,
            c.section_header,
            q.question_type,
            q.difficulty,
            q.question_text,
            q.answer_text
        FROM chunks c
        JOIN chunk_questions q ON c.chunk_id = q.chunk_id
        JOIN chunk_enrichments e ON c.chunk_id = e.chunk_id
        WHERE c.file_id = 'ae65897b-b0d0-47cf-990f-3c693a728b68'
        ORDER BY e.processed_at DESC;
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
