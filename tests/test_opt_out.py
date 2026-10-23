import os
import sys

import pytest

# Ensure project root is in sys.path when running pytest directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from graph import is_explicit_opt_out


@pytest.mark.parametrize(
    "reply",
    [
        # Standalone single-token opt-outs
        "STOP",
        "stop.",
        "STOP!!!",
        "UNSUBSCRIBE",
        "unsubscribe please",
        "please unsubscribe",
        "unsubscribe me",
        "opt out",
        "opt-out",
        "stop please",
        # List removal requests
        "Please remove me from your list.",
        "Take me off your mailing list.",
        "remove me from this list please",
        "take me off the email list",
        "delete me from your list",
        "delete me from your database",
        "delete me from your records",
        # Cease-contact directives
        "Don't contact me again.",
        "Do not email me.",
        "Stop emailing me.",
        "Stop contacting me.",
        "Stop messaging",
        "No further contact, please.",
        "no further emails",
        "no further messages",
        "don't follow up with me again",
        # Hybrid OOO + explicit opt-out
        "Out of office. Back Tuesday. Please don't follow up.",
    ],
)
def test_explicit_opt_out_detected(reply: str):
    """Ensure explicit opt-out and unsubscribe commands evaluate to True."""
    assert is_explicit_opt_out(reply) is True


@pytest.mark.parametrize(
    "reply",
    [
        # Soft rejections / objections (Not legal opt-outs)
        "Not interested, thanks.",
        "No.",
        "We already have an in-house video team, so we're good for now.",
        "We work with a big agency on retainer. Not looking to add anyone.",
        "We're shutting down the brand this month unfortunately, so not the right time.",
        # Temporal deferrals / follow-up timing constraints (NOT opt-outs)
        "stop emailing me for now.",
        "stop emailing me for now",
        "stop emailing for now",
        "stop contacting me for now",
        "stop messaging me right now",
        "don't contact me for now",
        "please don't follow up with me for now",
        "don't contact me at this time",
        "Don't follow up until next month.",
        "Don't contact me before Monday.",
        "Don't email me this week; try me in January.",
        "Don't follow up for now.",
        "Please don't follow up until we speak with our board",
        "Don't contact me unless you have case studies",
        "don't email me after 5pm",
        "Busy this month, try me in Jan.",
        # Channel redirection (not an unsubscribe)
        "Please don't follow up on WhatsApp; email instead.",
        "Don't contact me on LinkedIn, send an email to info@example.com",
        # Negation / keeping on list
        "Don't remove me from your list.",
        # Positive replies, inquiries, and conversational text
        "Sure, send them over.",
        "What's the pricing like?",
        "Who is this?",
        "haan bhai bhejo, dekhte hain",
        # Empty and minimal non-verbal input
        "",
        "   ",
        ".",
        "👍",
    ],
)
def test_non_opt_out_replies(reply: str):
    """Ensure soft rejections, temporal deferrals, and other non-opt-outs evaluate to False."""
    assert is_explicit_opt_out(reply) is False
