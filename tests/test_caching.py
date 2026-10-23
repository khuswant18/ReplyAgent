import os
import sys
from pathlib import Path
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cache import ClassificationCache, make_cache_key, reset_stats, get_stats
from schemas import ReplyAnalysis
from graph import get_client, max_attempt, classify_with_groq


def test_sqlite_cache_set_and_get(tmp_path: Path):
    db_file = tmp_path / "test_cache.sqlite3"
    test_cache = ClassificationCache(db_path=db_file)

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
    assert test_cache.get(key) is None

    test_cache.set(key, sample_analysis)
    retrieved = test_cache.get(key)
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

    # 4. Changed schema
    diff_schema = {"type": "object", "properties": {"foo": {"type": "string"}}}
    assert make_cache_key(context_base, model_base, prompt_base, schema=diff_schema) != key_base

    # 5. Changed cache version
    assert make_cache_key(context_base, model_base, prompt_base, cache_version="v2") != key_base


def test_groq_sdk_retries_disabled():
    client = get_client()
    assert client.max_retries == 0


def test_application_retries_bounded():
    assert max_attempt == 2
