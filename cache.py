import os
import json
import sqlite3
import hashlib
from pathlib import Path
from schemas import ReplyAnalysis

DB_PATH = Path(
    os.getenv("REPLY_CACHE_DB_PATH", "output/reply_cache.sqlite3")
)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

CACHE_VERSION = "1"

to_show = {
    "api_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}


def reset_stats():
    for key in to_show:
        to_show[key] = 0


def get_stats():
    return to_show.copy()


def init_db():
    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            cache_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
        )
    """)

    connection.commit() #save database changes
    connection.close() 


def make_cache_key(context, model_name, system_prompt):
    data = {
        "version": CACHE_VERSION,
        "context": context,
        "model": model_name,
        "prompt": system_prompt,
        "schema": ReplyAnalysis.model_json_schema(),
    }

    text = json.dumps( #convert from dict to json 
        data,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(text.encode("utf-8")).hexdigest() #generate a unique sha key i.e 64 hexadecimal character


def get_cached_analysis(cache_key):
    conn = sqlite3.connect(DB_PATH, timeout=30)

    row = conn.execute(
        "SELECT result_json FROM classifications WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()

    conn.close()

    if row is None:
        return None

    try:
        return ReplyAnalysis.model_validate_json(row[0])

    except (ValueError, TypeError):
        conn = sqlite3.connect(DB_PATH, timeout=30)

        conn.execute(
            "DELETE FROM classifications WHERE cache_key = ?",
            (cache_key,),
        )

        conn.commit()
        conn.close()

        return None


def save_cached_analysis(cache_key, analysis):
    conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.execute(
        """
        INSERT INTO classifications (cache_key, result_json)
        VALUES (?, ?)
        ON CONFLICT(cache_key)
        DO UPDATE SET result_json = excluded.result_json
        """,
        (cache_key, analysis.model_dump_json()),
    )

    conn.commit()
    conn.close()


init_db()