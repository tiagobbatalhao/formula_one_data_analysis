import os
from typing import List

import psycopg2
# Assuming this script is intended to be run from the the `src` directory or
# with `src` in your PYTHONPATH.
from models import SessionMetadata
from psycopg2.extras import execute_values


def insert_session_metadata_batch(
    conn, sessions: List[SessionMetadata], page_size: int = 1000
):
    """
    Perform a batch insert of SessionMetadata Pydantic models into the Postgres table.
    Uses psycopg2.extras.execute_values for efficient batch insertion.
    """
    if not sessions:
        return

    # Extract field names from the Pydantic v2 model
    columns = list(SessionMetadata.model_fields.keys())

    # Create the column string for the SQL query
    columns_str = ", ".join(columns)

    # The ON CONFLICT DO NOTHING requires the session_id to be a UNIQUE constraint or PRIMARY KEY
    query = f"""
        INSERT INTO session_metadata ({columns_str})
        VALUES %s
        ON CONFLICT (session_id) DO NOTHING;
    """

    # Prepare values as a list of tuples
    values = []
    for session in sessions:
        dump = session.model_dump()
        row = tuple(dump[col] for col in columns)
        values.append(row)

    with conn.cursor() as cur:
        # execute_values is much faster than executemany for bulk inserts
        execute_values(cur, query, values, page_size=page_size)

    conn.commit()


if __name__ == "__main__":
    # Example execution testing logic
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME", "f1_database")
    DB_USER = os.environ.get("DB_USER", "myuser")
    DB_PASS = os.environ.get("DB_PASS", "mypassword")

    conn_params = {
        "host": DB_HOST,
        "port": DB_PORT,
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASS,
    }

    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**conn_params)

        # Example dummy list (you will populate this list using your actual F1 data ingestion mechanism)
        sample_sessions: List[SessionMetadata] = []

        # insert_session_metadata_batch(conn, sample_sessions)

        print("Batch insert successfully executed.")

    except Exception as e:
        print(f"Failed to connect or insert data: {e}")
    finally:
        if "conn" in locals():
            conn.close()
