"""SQL tool for querying databases."""

import os
import sqlite3

import boto3

from rnow.core import ToolArgs, tool

CACHE = "/tmp/skyrl_dbs"
os.makedirs(CACHE, exist_ok=True)


@tool
def sql(args: ToolArgs, query: str) -> str:
    """Execute SQL query against the database."""
    db_id = args.metadata["db_id"]
    bucket = os.environ["S3_BUCKET"]
    path = f"{CACHE}/{db_id}.sqlite"

    try:
        if not os.path.exists(path):
            boto3.client("s3").download_file(bucket, f"dbs/{db_id}.sqlite", path)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(query).fetchall()
        conn.close()
        return f"{len(rows)} rows" if rows else "Empty"
    except Exception as e:
        return f"Error: {e}"
