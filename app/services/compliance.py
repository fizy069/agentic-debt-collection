"""Deterministic compliance guardrails for debt-collection agents.

  2. No false threats — only allowed consequences.
  3. Stop-contact — detect, acknowledge, halt pipeline.
  4. Offers within policy-defined ranges.
  5. Hardship/crisis — detect and flag for routing.
  7. Professional language — detect abusive borrower input.
  8. Privacy — redact sensitive identifiers from inputs and outputs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule 8 — Privacy / PII redaction
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(
    r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
)

_FULL_ACCOUNT_PATTERN = re.compile(
    r"\b(?:account\s*(?:#|number|num|no)?\.?\s*:?\s*)(\d{5,})\b",
    re.IGNORECASE,
)

_CREDIT_CARD_PATTERN = re.compile(
    r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
)

PII_REDACTION_MARKER = "[REDACTED]"


def redact_pii(text: str) -> tuple[str, bool]:
    """Replace full SSNs, long account numbers, and credit card numbers.

    Returns (cleaned_text, was_redacted).
    """
    cleaned = text
    redacted = False

    for pattern in (_SSN_PATTERN, _FULL_ACCOUNT_PATTERN, _CREDIT_CARD_PATTERN):
        if pattern.search(cleaned):
            cleaned = pattern.sub(PII_REDACTION_MARKER, cleaned)
            redacted = True

    return cleaned, redacted


# ---------------------------------------------------------------------------
# Rule 3 — Stop-contact detection
# ---------------------------------------------------------------------------

_STOP_CONTACT_PHRASES = (
    "stop contacting",
    "stop calling",
    "do not contact",
    "don't contact",
    "cease contact",
    "cease communication",
    "stop all contact",
    "remove me",
    "take me off",
    "no more calls",
    "no more contact",
    "leave me alone",
    "stop reaching out",
    "don't call me",
    "do not call",
)

STOP_CONTACT_REPLY = (
    "I understand your request to stop contact. Your request has been acknowledged "
    "and your account has been flagged accordingly. No further outreach will be "
    "initiated through this channel. If you wish to resume communication or explore "
    "your options in the future, please contact us directly."
)


def detect_stop_contact(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _STOP_CONTACT_PHRASES)


# ---------------------------------------------------------------------------
# Rule 5 — Hardship / crisis detection
# ---------------------------------------------------------------------------

_HARDSHIP_PHRASES = (
    "hardship",
    "financial hardship",
    "crisis",
    "can't afford",
    "cannot afford",
    "lost my job",
    "unemployed",
    "medical emergency",
    "disability",
    "homeless",
    "facing eviction",
    "bankruptcy",
)


def detect_hardship(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _HARDSHIP_PHRASES)


# ---------------------------------------------------------------------------
# Rule 7 — Abusive language detection
# ---------------------------------------------------------------------------

_ABUSIVE_PATTERNS = (
    r"\bf+u+c+k+",
    r"\bs+h+i+t+",
    r"\ba+s+s+h+o+l+e+",
    r"\bb+i+t+c+h+",
    r"\bd+a+m+n+\s+you",
    r"\bgo\s+to\s+hell\b",
    r"\bi\s*('ll|will)\s+(kill|hurt|find)\s+you",
    r"\bthreat",
    r"\bdie\b",
    r"\bkill\s+(you|your)",
)
_ABUSIVE_RE = re.compile("|".join(_ABUSIVE_PATTERNS), re.IGNORECASE)

ABUSIVE_CLOSE_REPLY = (
    "I understand this is a difficult situation, but I'm unable to continue this "
    "conversation while the language remains abusive. For your reference, your "
    "account details and options remain available. Please contact us again when "
    "you are ready to discuss next steps."
)


def detect_abusive(text: str) -> bool:
    return bool(_ABUSIVE_RE.search(text))


# ---------------------------------------------------------------------------
# Rule 4 — Offer / policy bounds (externalized)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OfferPolicy:
    """Concrete numeric bounds for debt-resolution offers."""

    min_settlement_pct: float
    max_settlement_pct: float
    min_plan_months: int
    max_plan_months: int


OFFER_POLICY = OfferPolicy(
    min_settlement_pct=40.0,
    max_settlement_pct=80.0,
    min_plan_months=3,
    max_plan_months=24,
)

_PERCENTAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MONTH_PATTERN = re.compile(
    r"(\d+)\s*[-\s]?\s*months?",
    re.IGNORECASE,
)


def check_offer_bounds(text: str, policy: OfferPolicy = OFFER_POLICY) -> list[str]:
    """Return list of violation descriptions found in assistant output."""
    violations: list[str] = []

    for match in _PERCENTAGE_PATTERN.finditer(text):
        value = float(match.group(1))
        if value < policy.min_settlement_pct or value > policy.max_settlement_pct:
            violations.append(
                f"Settlement percentage {value}% outside allowed range "
                f"[{policy.min_settlement_pct}%-{policy.max_settlement_pct}%]"
            )

    for match in _MONTH_PATTERN.finditer(text):
        months = int(match.group(1))
        if months < policy.min_plan_months or months > policy.max_plan_months:
            violations.append(
                f"Payment plan of {months} months outside allowed range "
                f"[{policy.min_plan_months}-{policy.max_plan_months} months]"
            )

    return violations


def offer_policy_directive(policy: OfferPolicy = OFFER_POLICY) -> str:
    """Produce a concrete directive string to inject into the LLM prompt.

    Reads the template from the centralized prompt registry so the
    self-learning loop can modify the wording without code changes.
    Falls back to an inline template if the registry is unavailable.
    """
    try:
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        section = registry.get_section("compliance_directives:offer_policy")
        return section.content.format(
            min_settlement_pct=policy.min_settlement_pct,
            max_settlement_pct=policy.max_settlement_pct,
            min_plan_months=policy.min_plan_months,
            max_plan_months=policy.max_plan_months,
        )
    except Exception:
        return (
            f"Settlement offers must be between {policy.min_settlement_pct}% and "
            f"{policy.max_settlement_pct}% of the outstanding balance. "
            f"Payment plans must be between {policy.min_plan_months} and "
            f"{policy.max_plan_months} months."
        )


# ---------------------------------------------------------------------------
# Rule 2 — Allowed consequences (externalized)
# ---------------------------------------------------------------------------

ALLOWED_CONSEQUENCES: tuple[str, ...] = (
    "referral to collections agency",
    "credit reporting",
    "legal review",
    "account charge-off",
)

_FALSE_THREAT_PHRASES = (
    "arrest",
    "jail",
    "prison",
    "garnish your wages",
    "garnishment",
    "sue you",
    "we will sue",
    "criminal charges",
    "warrant",
)


def check_false_threats(text: str) -> list[str]:
    """Return list of false-threat phrases found in assistant output."""
    lowered = text.lower()
    return [
        phrase for phrase in _FALSE_THREAT_PHRASES
        if phrase in lowered
    ]


def allowed_consequences_directive() -> str:
    """Produce a concrete directive listing permitted consequences.

    Reads the template from the centralized prompt registry so the
    self-learning loop can modify the wording without code changes.
    Falls back to an inline template if the registry is unavailable.
    """
    items = ", ".join(ALLOWED_CONSEQUENCES)
    try:
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        section = registry.get_section("compliance_directives:allowed_consequences")
        return section.content.format(consequences=items)
    except Exception:
        return (
            f"The only consequences you may reference are: {items}. "
            "Do not invent, imply, or state any other consequences."
        )
