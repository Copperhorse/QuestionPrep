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

    DATABASE_URL = "sqlite:////home/copper/Desktop/Project/QuestionPrep/data/rag_staging.db"
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
        LIMIT
        	3
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
        SELECT * FROM chunk_questions
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


@app.cell
def _(chunk_questions, chunks, engine, mo):
    _df = mo.sql(
        f"""
        SELECT q.question_id, q.question_text, q.answer_text,c.content,
               q.difficulty, q.question_type
        FROM chunk_questions q
        JOIN chunks c ON q.chunk_id = c.chunk_id
        WHERE c.should_use = 1
        """,
        engine=engine
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
