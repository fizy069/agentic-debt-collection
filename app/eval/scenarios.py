"""Borrower persona definitions and reproducible scenario generation.

Five persona types model the diversity of real borrower behaviour.  Each
persona carries an LLM system prompt that instructs the borrower-simulator
to play that role during simulated conversations.

``build_scenario_batch`` produces a deterministic set of scenarios for a
given seed so that evaluation runs are fully reproducible.
"""

from __future__ import annotations

import random
from typing import Sequence

from app.eval.models import BorrowerPersona, PersonaType, Scenario

# ------------------------------------------------------------------
# Persona definitions
# ------------------------------------------------------------------

PERSONAS: dict[PersonaType, BorrowerPersona] = {
    PersonaType.COOPERATIVE: BorrowerPersona(
        persona_type=PersonaType.COOPERATIVE,
        description="Willing borrower who confirms identity, acknowledges debt, and explores payment options.",
        system_prompt=(
            "You are playing a cooperative borrower in a debt-collection simulation. "
            "You received a letter about an overdue balance and want to resolve it.\n\n"
            "Behaviour rules:\n"
            "- Confirm your identity when asked (last four of account, date of birth).\n"
            "- Acknowledge the debt and express willingness to discuss options.\n"
            "- Ask reasonable questions about settlement or payment plans.\n"
            "- Stay polite and cooperative throughout.\n"
            "- Keep responses to 1-3 sentences."
        ),
    ),
    PersonaType.COMBATIVE: BorrowerPersona(
        persona_type=PersonaType.COMBATIVE,
        description="Hostile borrower who disputes the debt, argues, and pushes back aggressively.",
        system_prompt=(
            "You are playing a combative borrower in a debt-collection simulation. "
            "You are angry about being contacted and dispute the debt.\n\n"
            "Behaviour rules:\n"
            "- Challenge the agent's authority and the validity of the debt.\n"
            "- Express frustration and argue, but do NOT use profanity or "
            "slurs (stay just below the abuse threshold).\n"
            "- Refuse to confirm identity initially; relent partially after pressure.\n"
            "- Eventually, grudgingly engage with the process.\n"
            "- Keep responses to 1-3 sentences."
        ),
    ),
    PersonaType.EVASIVE: BorrowerPersona(
        persona_type=PersonaType.EVASIVE,
        description="Deflecting borrower who gives vague answers and avoids commitment.",
        system_prompt=(
            "You are playing an evasive borrower in a debt-collection simulation. "
            "You want to avoid committing to anything.\n\n"
            "Behaviour rules:\n"
            "- Give vague or non-committal answers ('maybe', 'I'll think about it').\n"
            "- Deflect direct questions about your finances.\n"
            "- Partially confirm identity but avoid discussing the debt amount.\n"
            "- Do not outright refuse; just keep hedging.\n"
            "- Keep responses to 1-3 sentences."
        ),
    ),
    PersonaType.CONFUSED: BorrowerPersona(
        persona_type=PersonaType.CONFUSED,
        description="Borrower who does not understand the debt, asks for clarification, and misinterprets offers.",
        system_prompt=(
            "You are playing a confused borrower in a debt-collection simulation. "
            "You don't understand why you owe money or what the process involves.\n\n"
            "Behaviour rules:\n"
            "- Ask what the debt is for; you are unsure which account it relates to.\n"
            "- Misunderstand terms (e.g., confuse 'settlement' with 'full payment').\n"
            "- Confirm identity slowly, needing clarification on what info is needed.\n"
            "- Eventually start to understand once the agent explains clearly.\n"
            "- Keep responses to 1-3 sentences."
        ),
    ),
    PersonaType.DISTRESSED: BorrowerPersona(
        persona_type=PersonaType.DISTRESSED,
        description="Borrower facing financial hardship or crisis who is emotionally vulnerable.",
        system_prompt=(
            "You are playing a distressed borrower in a debt-collection simulation. "
            "You recently lost your job and are facing financial hardship.\n\n"
            "Behaviour rules:\n"
            "- Mention your hardship early: you lost your job / have medical bills.\n"
            "- Express emotional distress but remain civil.\n"
            "- Confirm identity when asked.\n"
            "- Acknowledge the debt but explain you cannot afford to pay right now.\n"
            "- Ask about hardship programmes or reduced payment options.\n"
            "- Keep responses to 1-3 sentences."
        ),
    ),
}


# ------------------------------------------------------------------
# Scenario fixture data
# ------------------------------------------------------------------

_BORROWER_PROFILES: list[dict[str, object]] = [
    {
        "debt_amount": 4500.00,
        "days_past_due": 45,
        "notes": "Credit card balance, minimum payments missed.",
    },
    {
        "debt_amount": 12000.00,
        "days_past_due": 90,
        "notes": "Personal loan, 3 months delinquent.",
    },
    {
        "debt_amount": 2200.50,
        "days_past_due": 30,
        "notes": "Medical bill, recently defaulted.",
    },
    {
        "debt_amount": 8750.00,
        "days_past_due": 120,
        "notes": "Auto loan deficiency balance after repossession.",
    },
    {
        "debt_amount": 950.00,
        "days_past_due": 60,
        "notes": "Utility bill sent to collections.",
    },
]

_INITIAL_MESSAGES: dict[PersonaType, str] = {
    PersonaType.COOPERATIVE: "Hi, I got a letter about an overdue balance. I'd like to understand my options.",
    PersonaType.COMBATIVE: "Why are you contacting me? I don't owe anything.",
    PersonaType.EVASIVE: "Uh, I got something in the mail. What's this about?",
    PersonaType.CONFUSED: "Hello? I received a notice but I'm not sure what it's about.",
    PersonaType.DISTRESSED: "I got your letter. I lost my job recently and I'm struggling financially.",
}


# ------------------------------------------------------------------
# Batch builder
# ------------------------------------------------------------------

def build_scenario_batch(
    seed: int = 42,
    n_per_persona: int = 2,
    persona_types: Sequence[PersonaType] | None = None,
) -> list[Scenario]:
    """Generate a reproducible batch of evaluation scenarios.

    Args:
        seed: RNG seed for deterministic generation.
        n_per_persona: Number of scenarios to create per persona type.
        persona_types: Subset of personas to include.  Defaults to all five.

    Returns:
        Flat list of ``Scenario`` objects, sorted by persona then index.
    """
    rng = random.Random(seed)
    types = list(persona_types or PersonaType)
    scenarios: list[Scenario] = []

    for ptype in types:
        persona = PERSONAS[ptype]
        for i in range(n_per_persona):
            profile = _BORROWER_PROFILES[rng.randint(0, len(_BORROWER_PROFILES) - 1)]
            borrower_id = f"eval-{ptype.value}-{i:02d}"
            scenarios.append(
                Scenario(
                    scenario_id=f"{ptype.value}-{seed}-{i:02d}",
                    persona=persona,
                    borrower_id=borrower_id,
                    account_reference=f"ACCT{rng.randint(100000, 999999)}",
                    debt_amount=float(profile["debt_amount"]),
                    currency="USD",
                    days_past_due=int(profile["days_past_due"]),
                    borrower_message=_INITIAL_MESSAGES[ptype],
                    notes=str(profile.get("notes", "")),
                    seed=seed,
                )
            )

    return scenarios
