from graph import reply_graph
from schemas import ReplyAnalysis


def analyze_reply(row: dict[str, str]) -> ReplyAnalysis:
    """Analyze a single sales reply using the production LangGraph workflow."""
    result = reply_graph.invoke(
        {
            "row": row,
            "attempts": 0,
        }
    )

    analysis = result.get("analysis")
    if analysis is None:
        raise ValueError("Graph did not return an analysis.")

    return analysis


if __name__ == "__main__":
    sample_row = {
        "id": "1",
        "received_at": "2026-10-01 09:12",
        "sender_name": "Aarav Mehta",
        "sender_title": "Founder",
        "company": "Brewbox Coffee",
        "our_message": "Hi Aarav, we make short video ads for D2C brands. Open to a quick look at a few we did for coffee brands?",
        "their_reply": "Sure, send them over. We're planning Diwali creatives right now actually.",
    }

    print("Running sample reply triage via main.py:")
    result = analyze_reply(sample_row)
    print(result.model_dump_json(indent=2))
