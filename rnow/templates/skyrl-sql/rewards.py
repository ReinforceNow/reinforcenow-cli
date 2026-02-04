"""SkyRL-style SQL rewards."""

import os
import re
import sqlite3
from collections import Counter

import boto3

from rnow.core import RewardArgs, get_response, reward

CACHE = "/tmp/skyrl_dbs"
os.makedirs(CACHE, exist_ok=True)


@reward
def format(args: RewardArgs, messages: list) -> float:
    response = get_response(messages)
    has_solution = bool(re.search(r"<solution>.*?</solution>", response, re.DOTALL | re.IGNORECASE))
    has_tool_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
    return 1.0 if has_solution and has_tool_call else 0.0


@reward
def sql_execution(args: RewardArgs, messages: list) -> float:
    response = get_response(messages)
    match = re.search(r"<solution>(.*?)</solution>", response, re.DOTALL | re.IGNORECASE)
    if not match:
        return 0.0

    generated = match.group(1).strip()
    db_id, expected = args.metadata["db_id"], args.metadata["expected_sql"]
    path = f"{CACHE}/{db_id}.sqlite"

    try:
        if not os.path.exists(path):
            boto3.client("s3").download_file(args.secrets["S3_BUCKET"], f"dbs/{db_id}.sqlite", path)
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        result = (
            1.0
            if Counter(conn.execute(generated).fetchall())
            == Counter(conn.execute(expected).fetchall())
            else 0.0
        )
        conn.close()
        return result
    except:
        return 0.0
