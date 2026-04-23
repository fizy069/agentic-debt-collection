from __future__ import annotations

import json
from datetime import date

from app.activities.agents import (
    _STAGE_FIELD_KEYS,
    _TURN_CAPS,
    _final_notice_expiry_date,
    _parse_stage_response,
)
from app.models.pipeline import LLMStageResponse, PipelineStage
from app.services.prompt_assembler import _render_turn_directives
from app.services.prompt_registry import get_prompt_registry, reset_prompt_registry


def _get_turn_directives(stage: PipelineStage, final_notice_expiry: str | None = None) -> str:
    """Helper that loads the registry config and renders turn directives."""
    reset_prompt_registry()
    registry = get_prompt_registry()
    config = registry.get_agent_config(stage)
    return _render_turn_directives(config, final_notice_expiry)


def test_final_notice_expiry_date_uses_7_day_window():
    assert _final_notice_expiry_date(base_date=date(2026, 4, 21)) == "2026-04-28"


def test_final_notice_directives_include_concrete_expiry_and_no_placeholder():
    directives = _get_turn_directives(
        PipelineStage.FINAL_NOTICE,
        final_notice_expiry="2026-04-28",
    )

    assert "Use this exact hard expiry date: 2026-04-28." in directives
    assert "Never use placeholders" in directives


def test_final_notice_directives_include_closing_guidance():
    directives = _get_turn_directives(
        PipelineStage.FINAL_NOTICE,
        final_notice_expiry="2026-04-28",
    )

    assert "close with a definitive statement" in directives
    assert "acknowledgement question" in directives


def test_non_final_stage_directives_include_follow_up_and_closing_guidance():
    directives = _get_turn_directives(PipelineStage.RESOLUTION)

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


# --- Prompt registry tests ---


def test_registry_loads_all_system_templates():
    reset_prompt_registry()
    registry = get_prompt_registry()
    for stage in (PipelineStage.ASSESSMENT, PipelineStage.RESOLUTION, PipelineStage.FINAL_NOTICE):
        section = registry.get_section(f"system_template:{stage.value}")
        assert len(section.content) > 100
        assert section.version == "v1"


def test_registry_agent_config_includes_system_template_and_directives():
    reset_prompt_registry()
    registry = get_prompt_registry()
    config = registry.get_agent_config(PipelineStage.RESOLUTION)
    assert "system_template" in config.sections
    assert "turn_directives" in config.sections
    assert "offer_policy" in config.sections
    assert "allowed_consequences" in config.sections


def test_registry_assessment_config_has_no_compliance_directives():
    reset_prompt_registry()
    registry = get_prompt_registry()
    config = registry.get_agent_config(PipelineStage.ASSESSMENT)
    assert "system_template" in config.sections
    assert "turn_directives" in config.sections
    assert "offer_policy" not in config.sections
    assert "allowed_consequences" not in config.sections


def test_registry_override_and_rollback():
    reset_prompt_registry()
    registry = get_prompt_registry()
    key = "turn_directives:default"
    original = registry.get_section(key)

    registry.override_section(key, "new content", "v2-exp")
    updated = registry.get_section(key)
    assert updated.content == "new content"
    assert updated.version == "v2-exp"

    registry.rollback(key, original.version)
    restored = registry.get_section(key)
    assert restored.content == original.content
    assert restored.version == original.version


def test_registry_snapshot_contains_all_sections():
    reset_prompt_registry()
    registry = get_prompt_registry()
    snap = registry.snapshot()
    assert "system_template:assessment" in snap
    assert "judge_system" in snap
    assert "overflow_system" in snap
    assert "compliance_directives:offer_policy" in snap


def test_config_version_changes_on_override():
    reset_prompt_registry()
    registry = get_prompt_registry()
    config_before = registry.get_agent_config(PipelineStage.ASSESSMENT)
    v_before = config_before.version

    registry.override_section(
        "system_template:assessment", "modified", "v2",
    )
    config_after = registry.get_agent_config(PipelineStage.ASSESSMENT)
    assert config_after.version != v_before
