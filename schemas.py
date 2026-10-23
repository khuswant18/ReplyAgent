from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


Disposition = Literal["yes", "no", "future", "needs_review"]

Intent = Literal[
    "interested",
    "pricing_question",
    "request_for_examples",
    "objection",
    "referral",
    "wrong_contact",
    "follow_up_later",
    "out_of_office",
    "unsubscribe",
    "not_interested",
    "ambiguous",
    "other",
]

Priority = Literal["high", "medium", "low"]

Sentiment = Literal["positive", "neutral", "negative", "mixed"]

NextAction = Literal[
    "offer_call",
    "send_examples",
    "answer_pricing",
    "answer_question",
    "contact_referral",
    "update_contact",
    "schedule_follow_up",
    "wait_until_return",
    "suppress_contact",
    "no_action",
    "human_review",
]


class ReplyAnalysis(BaseModel):
    """Structured analysis of a single sales reply."""

    model_config = ConfigDict(extra="forbid")

    disposition: Disposition = Field(
        description="Broad outcome: yes, no, future, or needs_review."
    )

    intent: Intent = Field(
        description="The primary intent of the customer's reply."
    )

    secondary_intents: list[Intent] = Field(
        description="Other intents present, excluding the primary intent."
    )

    priority: Priority = Field(
        description="How urgently the sales team should handle this reply."
    )

    sentiment: Sentiment = Field(
        description="Overall sentiment expressed in the reply."
    )

    next_action: NextAction = Field(
        description="Recommended next business action."
    )

    follow_up_date: str | None = Field(
        description="Follow-up date in YYYY-MM-DD format, or null if unknown."
    )

    referral_name: str | None = Field(
        description="Referred person's name, or null if unavailable."
    )

    referral_email: str | None = Field(
        description="Referred person's email, or null if unavailable."
    )

    draft_reply: str | None = Field(
        description="Suggested response, or null when no reply is appropriate."
    )

    do_not_contact: bool = Field(
        description="True when the prospect explicitly requests no further contact."
    )

    needs_human_review: bool = Field(
        description="True when the result requires human verification."
    )

    review_reason: str | None = Field(
        description="Why human review is required, or null otherwise."
    )

    rationale: str = Field(
        description="Brief explanation of the classification and recommendation."
    )

    @field_validator("follow_up_date")
    @classmethod
    def validate_follow_up_date(cls, value: str | None) -> str | None:
        if value is None:
            return None

        try:
            parsed_date = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "follow_up_date must use YYYY-MM-DD format."
            ) from exc

        if parsed_date.isoformat() != value:
            raise ValueError(
                "follow_up_date must use YYYY-MM-DD format."
            )

        return value

    @model_validator(mode="after")
    def validate_business_rules(self):
        if self.do_not_contact:
            self.intent = "unsubscribe"
            self.disposition = "no"
            self.next_action = "suppress_contact"
            self.draft_reply = None
            self.needs_human_review = False
            self.review_reason = None

        if self.intent == "unsubscribe":
            self.do_not_contact = True
            self.disposition = "no"
            self.next_action = "suppress_contact"
            self.draft_reply = None
            self.needs_human_review = False
            self.review_reason = None

        if self.disposition == "needs_review":
            self.needs_human_review = True
            self.next_action = "human_review"
            if not self.review_reason:
                self.review_reason = "Flagged for manual review."
        else:
            self.needs_human_review = False
            self.review_reason = None

        return self