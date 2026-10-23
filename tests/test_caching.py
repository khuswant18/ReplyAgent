import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3
from contextlib import closing

from cache import (
    get_cached_analysis,
    save_cached_analysis,
    make_cache_key,
    reset_stats,
    get_stats,
)
from schemas import ReplyAnalysis
from graph import get_client, max_attempt, classify_with_groq


def test_sqlite_cache_set_and_get(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_file = tmp_path / "test_cache.sqlite3"
    monkeypatch.setattr("cache.DB_PATH", db_file)

    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
        """)
        conn.commit()

    sample_analysis = ReplyAnalysis(
        disposition="yes",
        intent="interested",
        secondary_intents=[],
        priority="high",
        sentiment="positive",
        next_action="offer_call",
        follow_up_date=None,
        referral_name=None,
        referral_email=None,
        draft_reply="Sounds great!",
        do_not_contact=False,
        needs_human_review=False,
        review_reason=None,
        rationale="Clear interest.",
    )

    key = "sample_key_123"
    assert get_cached_analysis(key) is None

    save_cached_analysis(key, sample_analysis)
    retrieved = get_cached_analysis(key)
    assert retrieved is not None
    assert retrieved.disposition == "yes"
    assert retrieved.intent == "interested"
    assert retrieved.draft_reply == "Sounds great!"


def test_cache_invalidation_triggers():
    context_base = {
        "received_at": "2026-10-01",
        "sender_name": "Alice",
        "their_reply": "Send pricing please",
    }
    model_base = "openai/gpt-oss-20b"
    prompt_base = "Analyze sales reply"

    key_base = make_cache_key(context_base, model_base, prompt_base)

    # 1. Changed reply
    context_diff = dict(context_base, their_reply="Not interested")
    assert make_cache_key(context_diff, model_base, prompt_base) != key_base

    # 2. Changed model
    assert make_cache_key(context_base, "llama-3.3-70b-versatile", prompt_base) != key_base

    # 3. Changed prompt
    assert make_cache_key(context_base, model_base, "Different prompt") != key_base


def test_stats_tracking():
    reset_stats()
    stats = get_stats()
    assert stats["api_calls"] == 0
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 0


def test_groq_sdk_retries_disabled():
    client = get_client()
    assert client.max_retries == 0


def test_application_retries_bounded():
    assert max_attempt == 2


def test_classify_with_groq_cache_hit_and_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import MagicMock, patch
    from graph import SYSTEM_PROMPT, model_name

    db_file = tmp_path / "test_cache2.sqlite3"
    monkeypatch.setattr("cache.DB_PATH", db_file)
    monkeypatch.setattr("graph.time.sleep", lambda s: None)

    with closing(sqlite3.connect(db_file)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS classifications (
                cache_key TEXT PRIMARY KEY,
                result_json TEXT NOT NULL
            )
        """)
        conn.commit()

    reset_stats()

    row = {
        "received_at": "2026-10-01",
        "sender_name": "Rohan",
        "sender_title": "Founder",
        "company": "Dairy",
        "our_message": "Hi",
        "their_reply": "Send details",
    }

    cached_analysis = ReplyAnalysis(
        disposition="yes",
        intent="interested",
        secondary_intents=[],
        priority="high",
        sentiment="positive",
        next_action="send_examples",
        follow_up_date=None,
        referral_name=None,
        referral_email=None,
        draft_reply="Here are details",
        do_not_contact=False,
        needs_human_review=False,
        review_reason=None,
        rationale="Cached reply",
    )

    context = {
        "received_at": row["received_at"],
        "sender_name": row["sender_name"],
        "sender_title": row["sender_title"],
        "company": row["company"],
        "our_message": row["our_message"],
        "their_reply": row["their_reply"],
    }
    key = make_cache_key(context, model_name, SYSTEM_PROMPT)
    save_cached_analysis(key, cached_analysis)

    # 1. With bypass_cache=False: returns cached analysis, increments cache_hits, no API call
    result = classify_with_groq(row, bypass_cache=False)
    assert result.rationale == "Cached reply"
    stats = get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 0
    assert stats["api_calls"] == 0

    # 2. With bypass_cache=True: bypasses cache read, calls Groq, bypasses cache write
    fresh_analysis = ReplyAnalysis(
        disposition="yes",
        intent="interested",
        secondary_intents=[],
        priority="medium",
        sentiment="neutral",
        next_action="offer_call",
        follow_up_date=None,
        referral_name=None,
        referral_email=None,
        draft_reply="Fresh call",
        do_not_contact=False,
        needs_human_review=False,
        review_reason=None,
        rationale="Fresh reply from Groq",
    )

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content=fresh_analysis.model_dump_json()))]
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("graph.get_client", return_value=mock_client):
        fresh_result = classify_with_groq(row, bypass_cache=True)

    assert fresh_result.rationale == "Fresh reply from Groq"
    stats = get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["api_calls"] == 1

    # Verify cache write was bypassed: cached item in database is still the old one!
    cached_in_db = get_cached_analysis(key)
    assert cached_in_db.rationale == "Cached reply"

