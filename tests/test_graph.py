from graph import reply_graph


def test_explicit_opt_out_skips_llm():
    row = {
        "id": "test-1",
        "received_at": "2026-10-23",
        "sender_name": "Test Prospect",
        "sender_title": "Marketing Manager",
        "company": "Example Company",
        "our_message": "Would you be interested in video ads?",
        "their_reply": "Please remove me from your list.",
    }

    result = reply_graph.invoke({
        "row": row,
        "attempts": 0,
    })

    analysis = result["analysis"]

    assert analysis.intent == "unsubscribe"
    assert analysis.do_not_contact is True
    assert analysis.next_action == "suppress_contact"
    assert analysis.draft_reply is None