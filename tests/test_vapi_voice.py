from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.activities.agents import _finalize_resolution_voice_call
from app.models.pipeline import (
    PipelineStage,
    ResolutionVoiceFinalizeInput,
    StageTurnOutput,
)
from app.services.vapi_client import resolve_resolution_voice_settings
from app.workflows import borrower_workflow as workflow_module
from app.workflows.borrower_workflow import BorrowerWorkflow


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _FakeWorkflowHandle:
    def __init__(self) -> None:
        self.signals: list[tuple[object, dict]] = []

    async def signal(self, signal_method: object, payload: dict) -> None:
        self.signals.append((signal_method, payload))


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.handle = _FakeWorkflowHandle()
        self.workflow_ids: list[str] = []

    def get_workflow_handle(self, workflow_id: str) -> _FakeWorkflowHandle:
        self.workflow_ids.append(workflow_id)
        return self.handle


def _borrower_payload() -> dict:
    return {
        "borrower": {
            "borrower_id": "b-vapi-001",
            "account_reference": "acct-0001",
            "debt_amount": 1200.0,
            "currency": "USD",
            "days_past_due": 45,
            "borrower_message": "I received the notice and need options.",
            "notes": "prefers voice for resolution",
        }
    }


def _stage_output(
    *,
    stage: str,
    channel: str = "chat",
    assistant_reply: str = "ok",
    stage_complete: bool = True,
    collected_fields: dict[str, bool] | None = None,
    decision: str = "done",
    transition_reason: str = "done",
    next_stage: str | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "stage": stage,
        "channel": channel,
        "assistant_reply": assistant_reply,
        "summary": assistant_reply,
        "decision": decision,
        "stage_complete": stage_complete,
        "collected_fields": collected_fields or {},
        "transition_reason": transition_reason,
        "next_stage": next_stage,
        "metadata": metadata or {},
        "compliance_flags": {
            "stop_contact_requested": False,
            "hardship_detected": False,
            "abusive_borrower": False,
        },
    }


class TestVoiceModeSelection:
    def test_stub_mode_selected(self):
        with patch.dict("os.environ", {"AGENT2_VOICE_MODE": "stub"}, clear=True):
            settings = resolve_resolution_voice_settings()
        assert settings.mode.value == "stub"
        assert settings.fallback_reason == "stub_mode_selected"

    def test_vapi_mode_falls_back_when_required_env_missing(self):
        with patch.dict("os.environ", {"AGENT2_VOICE_MODE": "vapi"}, clear=True):
            settings = resolve_resolution_voice_settings()
        assert settings.mode.value == "stub"
        assert settings.fallback_reason is not None
        assert "VAPI_API_KEY" in settings.fallback_reason
        assert "VAPI_WEBHOOK_BASE_URL" in settings.fallback_reason

    def test_vapi_mode_selected_when_required_env_present(self):
        with patch.dict(
            "os.environ",
            {
                "AGENT2_VOICE_MODE": "vapi",
                "VAPI_API_KEY": "test-key",
                "VAPI_WEBHOOK_BASE_URL": "https://example.ngrok.app",
            },
            clear=True,
        ):
            settings = resolve_resolution_voice_settings()
        assert settings.mode.value == "vapi"
        assert settings.provider is not None
        assert settings.webhook_base_url == "https://example.ngrok.app"


@pytest.mark.asyncio
async def test_workflow_resolution_vapi_path_waits_for_completion_signal(monkeypatch):
    activity_calls: list[str] = []

    async def fake_wait_condition(condition, timeout=None, timeout_summary=None):
        deadline = asyncio.get_running_loop().time() + (
            timeout.total_seconds() if timeout else 1.0
        )
        while not condition():
            if asyncio.get_running_loop().time() > deadline:
                raise asyncio.TimeoutError()
            await asyncio.sleep(0.01)

    async def fake_execute_activity(name, payload, **kwargs):
        activity_calls.append(name)
        if name == "assessment_agent":
            return _stage_output(
                stage="assessment",
                channel="chat",
                assistant_reply="Assessment complete.",
                stage_complete=True,
                decision="assessment_completed",
                transition_reason="required_assessment_fields_collected",
                next_stage="resolution",
                collected_fields={
                    "identity_confirmed": True,
                    "debt_acknowledged": True,
                    "ability_to_pay_discussed": True,
                },
            )
        if name == "create_resolution_voice_call":
            return {
                "provider": "vapi",
                "call_id": "call-123",
                "web_call_url": "https://calls.vapi.ai/join/call-123",
                "status": "queued",
                "metadata": {},
            }
        if name == "finalize_resolution_voice_call":
            return _stage_output(
                stage="resolution",
                channel="voice_web_call",
                assistant_reply="Resolution voice call completed.",
                stage_complete=True,
                decision="resolution_attempted",
                transition_reason="resolution_call_ended",
                next_stage="final_notice",
                collected_fields={
                    "options_reviewed": True,
                    "borrower_position_known": True,
                    "commitment_or_disposition": True,
                },
                metadata={"voice_transcript_excerpt": "borrower: I choose the plan."},
            )
        if name == "final_notice_agent":
            return _stage_output(
                stage="final_notice",
                channel="chat",
                assistant_reply="Final notice sent.",
                stage_complete=True,
                decision="final_notice_sent",
                transition_reason="final_notice_acknowledged",
                next_stage=None,
                collected_fields={
                    "final_notice_acknowledged": True,
                    "borrower_response_recorded": True,
                },
            )
        raise AssertionError(f"Unexpected activity: {name}")

    monkeypatch.setattr(
        workflow_module.workflow,
        "info",
        lambda: SimpleNamespace(workflow_id="wf-vapi-test"),
    )
    monkeypatch.setattr(
        workflow_module.workflow,
        "now",
        lambda: datetime(2026, 4, 22, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(workflow_module.workflow, "wait_condition", fake_wait_condition)
    monkeypatch.setattr(workflow_module.workflow, "execute_activity", fake_execute_activity)
    monkeypatch.setattr(workflow_module.workflow, "logger", _FakeLogger())

    workflow = BorrowerWorkflow()
    payload = _borrower_payload()
    payload["voice_settings"] = {
        "mode": "vapi",
        "provider": "vapi",
        "webhook_base_url": "https://example.ngrok.app",
    }

    run_task = asyncio.create_task(workflow.run(payload))

    for _ in range(200):
        await asyncio.sleep(0.01)
        status = workflow.get_status()
        if status["current_stage"] == PipelineStage.RESOLUTION.value:
            break
    else:
        raise AssertionError("Workflow never entered resolution stage.")

    status = workflow.get_status()
    assert status["resolution_voice_session"]["webCallUrl"] == "https://calls.vapi.ai/join/call-123"
    assert run_task.done() is False

    workflow.update_resolution_voice_event(
        {
            "event_type": "status-update",
            "workflow_id": "wf-vapi-test",
            "borrower_id": "b-vapi-001",
            "stage": "resolution",
            "provider": "vapi",
            "call_id": "call-123",
            "call_status": "in-progress",
            "metadata": {"source": "test"},
        }
    )
    await asyncio.sleep(0.05)
    assert run_task.done() is False

    workflow.update_resolution_voice_event(
        {
            "event_type": "end-of-call-report",
            "workflow_id": "wf-vapi-test",
            "borrower_id": "b-vapi-001",
            "stage": "resolution",
            "provider": "vapi",
            "call_id": "call-123",
            "call_status": "ended",
            "ended_reason": "customer-ended-call",
            "final_artifact_received": True,
            "artifact": {"summary": "Borrower agreed to a plan."},
            "messages": [
                {"role": "user", "content": "I choose a payment plan and I can pay monthly."},
                {"role": "assistant", "content": "Thanks, I have recorded your commitment."},
            ],
            "metadata": {"source": "test"},
        }
    )
    for _ in range(200):
        await asyncio.sleep(0.01)
        if workflow.get_status()["current_stage"] == PipelineStage.FINAL_NOTICE.value:
            workflow.add_borrower_message(
                {"message": "I acknowledge the final notice.", "message_id": "msg-final"}
            )
            break
    else:
        raise AssertionError("Workflow did not advance to final_notice after voice completion.")

    final_status = await asyncio.wait_for(run_task, timeout=3)
    assert final_status["completed"] is True
    assert final_status["current_stage"] == PipelineStage.COMPLETED.value
    assert "create_resolution_voice_call" in activity_calls
    assert "finalize_resolution_voice_call" in activity_calls
    resolution_outputs = [
        output for output in final_status["outputs"] if output["stage"] == "resolution"
    ]
    assert resolution_outputs
    assert resolution_outputs[-1]["channel"] == "voice_web_call"


def test_vapi_webhook_routes_signal(monkeypatch):
    fake_temporal_client = _FakeTemporalClient()

    async def fake_connect(*args, **kwargs):
        return fake_temporal_client

    monkeypatch.setattr(main_module.Client, "connect", fake_connect)
    monkeypatch.delenv("VAPI_WEBHOOK_AUTH_BEARER", raising=False)
    monkeypatch.delenv("VAPI_WEBHOOK_SECRET", raising=False)

    with TestClient(main_module.app) as client:
        status_payload = {
            "type": "status-update",
            "status": "in-progress",
            "call": {
                "id": "call-xyz",
                "metadata": {
                    "workflow_id": "wf-webhook-test",
                    "borrower_id": "b-vapi-001",
                    "stage": "resolution",
                },
            },
        }
        response = client.post("/webhooks/vapi", json=status_payload)
        assert response.status_code == 202
        assert response.json()["routed"] is True

        end_payload = {
            "type": "end-of-call-report",
            "call": {
                "id": "call-xyz",
                "status": "ended",
                "endedReason": "customer-ended-call",
                "metadata": {
                    "workflow_id": "wf-webhook-test",
                    "borrower_id": "b-vapi-001",
                    "stage": "resolution",
                },
                "artifact": {"summary": "Done"},
            },
            "messagesOpenAIFormatted": [
                {"role": "user", "content": "I choose the payment plan."},
                {"role": "assistant", "content": "I have recorded your commitment."},
            ],
        }
        response = client.post("/webhooks/vapi", json=end_payload)
        assert response.status_code == 202
        assert response.json()["routed"] is True

    assert fake_temporal_client.workflow_ids == ["wf-webhook-test", "wf-webhook-test"]
    assert len(fake_temporal_client.handle.signals) == 2
    signal_method, signal_payload = fake_temporal_client.handle.signals[-1]
    assert signal_method.__name__ == "update_resolution_voice_event"
    assert signal_payload["event_type"] == "end-of-call-report"
    assert signal_payload["final_artifact_received"] is True


@pytest.mark.asyncio
async def test_resolution_voice_finalization_returns_stage_turn_output():
    payload = ResolutionVoiceFinalizeInput(
        borrower={
            "borrower_id": "b-finalize",
            "account_reference": "acct-7777",
            "debt_amount": 2500.0,
            "currency": "USD",
            "days_past_due": 60,
            "borrower_message": "Initial borrower text.",
            "notes": None,
        },
        completed_stages=[],
        collected_fields={},
        turn_index=1,
        compliance_flags={"stop_contact_requested": False, "hardship_detected": False, "abusive_borrower": False},
        call_id="call-finalize",
        call_status="ended",
        ended_reason="customer-ended-call",
        artifact={"summary": "Borrower agreed to a payment plan."},
        messages=[
            {"role": "user", "content": "I choose the payment plan and I can pay monthly."},
            {"role": "assistant", "content": "Great, I have captured that commitment."},
        ],
        metadata={},
    ).model_dump(mode="json")

    result = await _finalize_resolution_voice_call(payload)
    turn = StageTurnOutput.model_validate(result)

    assert turn.stage == PipelineStage.RESOLUTION
    assert turn.channel.value == "voice_web_call"
    assert turn.stage_complete is True
    assert turn.next_stage == PipelineStage.FINAL_NOTICE
    assert turn.collected_fields["options_reviewed"] is True
    assert turn.collected_fields["borrower_position_known"] is True
    assert turn.collected_fields["commitment_or_disposition"] is True
    assert turn.metadata["voice_call_id"] == "call-finalize"
    assert turn.metadata["voice_final_artifact_received"] is True
