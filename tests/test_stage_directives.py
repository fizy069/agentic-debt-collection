from __future__ import annotations

from datetime import date

from app.activities.agents import _build_turn_directives, _final_notice_expiry_date
from app.models.pipeline import PipelineStage


def test_final_notice_expiry_date_uses_7_day_window():
    assert _final_notice_expiry_date(base_date=date(2026, 4, 21)) == "2026-04-28"


def test_final_notice_directives_include_concrete_expiry_and_no_placeholder():
    directives = _build_turn_directives(
        stage=PipelineStage.FINAL_NOTICE,
        stage_complete=False,
        final_notice_expiry="2026-04-28",
    )

    assert "Use this exact hard expiry date: 2026-04-28." in directives
    assert "Never use placeholders" in directives
    assert "Ask one concrete acknowledgement question" in directives


def test_final_notice_complete_directives_require_terminal_close():
    directives = _build_turn_directives(
        stage=PipelineStage.FINAL_NOTICE,
        stage_complete=True,
        final_notice_expiry="2026-04-28",
    )

    assert "Record the borrower's final response" in directives
    assert "Do not ask a follow-up question." in directives


def test_non_final_stage_directives_keep_follow_up_behavior():
    directives = _build_turn_directives(
        stage=PipelineStage.RESOLUTION,
        stage_complete=False,
    )

    assert "If stage is incomplete, ask one concrete follow-up question." in directives
    assert "If complete, provide a brief transition-ready response." in directives
