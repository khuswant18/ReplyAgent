import sqlite3
import os 
from pathlib import Path
from schemas import ReplyAnalysis
import json 
from pydantic import ValidationError
import hashlib

dbpath = Path(
    os.getenv("REPLY_CACHE_DB_PATH", "output/reply_cache.sqlite3")
)

stats = {
    "api_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}

to_show = stats


def reset_stats() -> None:
    for key in stats:
        stats[key] = 0


def get_stats() -> dict[str, int]:
    return stats.copy()


reset_to_show = reset_stats
get_to_show = get_stats


class ClassificationCache:
    def __init__(self, db_path: Path = dbpath):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS classifications (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, cache_key: str) -> ReplyAnalysis | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT result_json
                FROM classifications
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        try:
            # Validate cached data again before trusting it.
            return ReplyAnalysis.model_validate_json(row[0])
        except (ValidationError, ValueError, TypeError):
            # Remove corrupted or outdated entries; classify afresh.
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM classifications WHERE cache_key = ?",
                    (cache_key,),
                )
            return None

    def set(self, cache_key: str, analysis: ReplyAnalysis) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO classifications (cache_key, result_json)
                VALUES (?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result_json = excluded.result_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (cache_key, analysis.model_dump_json()),
            )


CACHE_VERSION = os.getenv("REPLY_CACHE_VERSION", "1")


def make_cache_key(
    context: dict,
    model_name: str,
    system_prompt: str,
    schema: dict | None = None,
    cache_version: str = CACHE_VERSION,
) -> str:
    payload = {
        "cache_version": cache_version,
        "context": context,
        "model": model_name,
        "system_prompt": system_prompt,
        "schema": schema if schema is not None else ReplyAnalysis.model_json_schema(),
    }

    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


cache = ClassificationCache()

