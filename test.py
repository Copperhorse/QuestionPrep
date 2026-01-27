import marimo

__generated_with = "0.18.4"
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

    DATABASE_URL = "sqlite:////home/copper/Desktop/Project/rag_staging.db"
    engine = sqlalchemy.create_engine(DATABASE_URL)
    return (engine,)


@app.cell
def _():
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
def _(chunks, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM chunks LIMIT 18
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
def _(chunk_enrichments, engine, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM chunk_enrichments
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
