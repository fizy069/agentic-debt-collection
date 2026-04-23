from __future__ import annotations

import json
from datetime import date

from app.activities.agents import (
    _STAGE_FIELD_KEYS,
    _TURN_CAPS,
    _build_turn_directives,
    _final_notice_expiry_date,
    _parse_stage_response,
)
from app.models.pipeline import LLMStageResponse, PipelineStage


def test_final_notice_expiry_date_uses_7_day_window():
    assert _final_notice_expiry_date(base_date=date(2026, 4, 21)) == "2026-04-28"


def test_final_notice_directives_include_concrete_expiry_and_no_placeholder():
    directives = _build_turn_directives(
        stage=PipelineStage.FINAL_NOTICE,
        final_notice_expiry="2026-04-28",
    )

    assert "Use this exact hard expiry date: 2026-04-28." in directives
    assert "Never use placeholders" in directives


def test_final_notice_directives_include_closing_guidance():
    directives = _build_turn_directives(
        stage=PipelineStage.FINAL_NOTICE,
        final_notice_expiry="2026-04-28",
    )

    assert "close with a definitive statement" in directives
    assert "acknowledgement question" in directives


def test_non_final_stage_directives_include_follow_up_and_closing_guidance():
    directives = _build_turn_directives(
        stage=PipelineStage.RESOLUTION,
    )

    assert "ask one concrete follow-up question" in directives
    assert "transition-ready closing statement" in directives


# --- _parse_stage_response tests ---


def _make_json_response(
    *,
    assistant_reply: str = "Hello borrower.",
    stage_complete: bool = False,
    collected_fields: dict | None = None,
    transition_reason: str = "awaiting_details",
    decision: str = "assessment_follow_up",
) -> str:
    return json.dumps({
        "assistant_reply": assistant_reply,
        "stage_complete": stage_complete,
        "collected_fields": collected_fields or {
            "identity_confirmed": False,
            "debt_acknowledged": False,
            "ability_to_pay_discussed": False,
        },
        "transition_reason": transition_reason,
        "decision": decision,
    })


def test_parse_stage_response_valid_json():
    raw = _make_json_response(
        assistant_reply="Thank you for confirming.",
        stage_complete=True,
        collected_fields={
            "identity_confirmed": True,
            "debt_acknowledged": True,
            "ability_to_pay_discussed": True,
        },
        transition_reason="all_fields_collected",
        decision="assessment_completed",
    )
    result = _parse_stage_response(
        raw,
        stage=PipelineStage.ASSESSMENT,
        collected_fields={},
    )

    assert isinstance(result, LLMStageResponse)
    assert result.assistant_reply == "Thank you for confirming."
    assert result.stage_complete is True
    assert result.collected_fields["identity_confirmed"] is True
    assert result.transition_reason == "all_fields_collected"
    assert result.decision == "assessment_completed"


def test_parse_stage_response_markdown_wrapped_json():
    inner = _make_json_response(assistant_reply="Wrapped reply.")
    raw = f"```json\n{inner}\n```"
    result = _parse_stage_response(
        raw,
        stage=PipelineStage.ASSESSMENT,
        collected_fields={},
    )

    assert result.assistant_reply == "Wrapped reply."
    assert result.stage_complete is False


def test_parse_stage_response_malformed_json_returns_safe_fallback():
    raw = "This is not JSON at all, just a plain text reply."
    prior_fields = {"identity_confirmed": True, "debt_acknowledged": False}
    result = _parse_stage_response(
        raw,
        stage=PipelineStage.ASSESSMENT,
        collected_fields=prior_fields,
    )

    assert result.assistant_reply == raw
    assert result.stage_complete is False
    assert result.transition_reason == "llm_response_parse_failed"
    assert result.collected_fields["identity_confirmed"] is True
    assert result.collected_fields["debt_acknowledged"] is False


def test_parse_stage_response_fallback_uses_stage_field_keys():
    raw = "Not JSON."
    result = _parse_stage_response(
        raw,
        stage=PipelineStage.RESOLUTION,
        collected_fields={"options_reviewed": True},
    )

    expected_keys = set(_STAGE_FIELD_KEYS[PipelineStage.RESOLUTION])
    assert set(result.collected_fields.keys()) == expected_keys
    assert result.collected_fields["options_reviewed"] is True
    assert result.collected_fields["borrower_position_known"] is False


def test_turn_caps_are_defined_for_all_active_stages():
    assert PipelineStage.ASSESSMENT in _TURN_CAPS
    assert PipelineStage.RESOLUTION in _TURN_CAPS
    assert PipelineStage.FINAL_NOTICE in _TURN_CAPS
    assert _TURN_CAPS[PipelineStage.FINAL_NOTICE] < _TURN_CAPS[PipelineStage.ASSESSMENT]
