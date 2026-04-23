"""Deterministic compliance guardrails for debt-collection agents.

  2. No false threats — only allowed consequences.
  3. Stop-contact — detect, acknowledge, halt pipeline.
  5. Hardship/crisis — detect and flag for routing.
  7. Professional language — detect abusive borrower input.
  8. Privacy — redact sensitive identifiers from inputs and outputs.

Rule 4 (offer bounds) is enforced via prompt directives and the LLM
compliance judge; a deterministic regex check was removed after it
produced unacceptable false-positive rates (e.g. "$200 monthly income"
matched as "200 months").
"""

from __future__ import annotations

import logging
import re

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
# Rule 4 — Offer policy (prompt-side guidance only; deterministic check removed)
# ---------------------------------------------------------------------------

OFFER_POLICY_BOUNDS = {
    "min_settlement_pct": 40.0,
    "max_settlement_pct": 80.0,
    "min_plan_months": 3,
    "max_plan_months": 24,
}


# ---------------------------------------------------------------------------
# Rule 2 — Allowed consequences (externalized)
# ---------------------------------------------------------------------------

ALLOWED_CONSEQUENCES: tuple[str, ...] = (
    "referral to collections agency",
    "credit reporting",
    "legal review",
    "account charge-off",
)

_FALSE_THREAT_PHRASES: tuple[str, ...] = (
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

_VERB_STEM_SUFFIX = r"(?:ed|ing|s)?"

_FALSE_THREAT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("arrest",           re.compile(r"\barrest" + _VERB_STEM_SUFFIX + r"\b", re.IGNORECASE)),
    ("jail",             re.compile(r"\bjail" + _VERB_STEM_SUFFIX + r"\b", re.IGNORECASE)),
    ("prison",           re.compile(r"\bprison\b", re.IGNORECASE)),
    ("garnish your wages", re.compile(r"\bgarnish(?:ed|ing|es)?\s+your\s+wages\b", re.IGNORECASE)),
    ("garnishment",      re.compile(r"\bgarnishments?\b", re.IGNORECASE)),
    ("sue you",          re.compile(r"\bsu(?:e|ed|ing)\s+you\b", re.IGNORECASE)),
    ("we will sue",      re.compile(r"\bwe\s+will\s+sue\b", re.IGNORECASE)),
    ("criminal charges", re.compile(r"\bcriminal\s+charges?\b", re.IGNORECASE)),
    ("warrant",          re.compile(r"\bwarrants?\b", re.IGNORECASE)),
)


def check_false_threats(text: str) -> list[str]:
    """Return list of false-threat phrases found in assistant output.

    Each phrase is matched as a whole-word token (allowing common verb
    stems where relevant) so that e.g. "issue you" does not match "sue
    you", and "reassurance" does not match "assurance".
    """
    return [
        phrase for phrase, pattern in _FALSE_THREAT_PATTERNS
        if pattern.search(text)
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
