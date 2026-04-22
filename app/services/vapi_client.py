from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

from app.models.pipeline import (
    BorrowerRequest,
    ConversationMessage,
    PipelineStage,
    ResolutionVoiceCallCreateInput,
    ResolutionVoiceCallCreateOutput,
    ResolutionVoiceEvent,
    ResolutionVoiceSettings,
    VoiceMode,
    VoiceProvider,
)
from app.services.handoff import build_handoff_summary
from app.services.prompt_loader import load_stage_prompt

logger = logging.getLogger(__name__)

_VAPI_API_BASE_URL = "https://api.vapi.ai"
_DEFAULT_MODEL_PROVIDER = "openai"
_DEFAULT_MODEL_NAME = "gpt-4o-mini"
_DEFAULT_VOICE_PROVIDER = "vapi"
_DEFAULT_VOICE_ID = "Elliot"
_DEFAULT_TRANSCRIBER_PROVIDER = "deepgram"


def _env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _normalize_voice_mode(raw: str | None) -> VoiceMode:
    if (raw or "").strip().lower() == VoiceMode.VAPI.value:
        return VoiceMode.VAPI
    return VoiceMode.STUB


def resolve_resolution_voice_settings() -> ResolutionVoiceSettings:
    """Resolve AGENT2 Resolution voice mode using env configuration."""
    requested_mode = _normalize_voice_mode(_env("AGENT2_VOICE_MODE"))
    webhook_base_url = _env("VAPI_WEBHOOK_BASE_URL")
    api_key_present = bool(_env("VAPI_API_KEY"))

    if requested_mode == VoiceMode.VAPI and (not api_key_present or not webhook_base_url):
        missing = []
        if not api_key_present:
            missing.append("VAPI_API_KEY")
        if not webhook_base_url:
            missing.append("VAPI_WEBHOOK_BASE_URL")
        return ResolutionVoiceSettings(
            mode=VoiceMode.STUB,
            provider=None,
            fallback_reason=f"missing_required_vapi_config:{','.join(missing)}",
        )

    if requested_mode == VoiceMode.STUB:
        return ResolutionVoiceSettings(
            mode=VoiceMode.STUB,
            provider=None,
            fallback_reason="stub_mode_selected",
        )

    return ResolutionVoiceSettings(
        mode=VoiceMode.VAPI,
        provider=VoiceProvider.VAPI,
        webhook_base_url=webhook_base_url.rstrip("/") if webhook_base_url else None,
        webhook_credential_id=_env("VAPI_WEBHOOK_CREDENTIAL_ID"),
        model_provider=_env("VAPI_MODEL_PROVIDER") or _DEFAULT_MODEL_PROVIDER,
        model_name=_env("VAPI_MODEL_NAME") or _DEFAULT_MODEL_NAME,
        voice_provider=_env("VAPI_VOICE_PROVIDER") or _DEFAULT_VOICE_PROVIDER,
        voice_id=_env("VAPI_VOICE_ID") or _DEFAULT_VOICE_ID,
        transcriber_provider=_env("VAPI_TRANSCRIBER_PROVIDER")
        or _DEFAULT_TRANSCRIBER_PROVIDER,
    )


def _borrower_snapshot(borrower: BorrowerRequest) -> str:
    masked_reference = borrower.account_reference[-4:]
    return (
        f"Borrower ID: {borrower.borrower_id}\n"
        f"Account Ref (last4): {masked_reference}\n"
        f"Debt: {borrower.debt_amount:.2f} {borrower.currency}\n"
        f"Days past due: {borrower.days_past_due}\n"
        f"Notes: {borrower.notes or 'None provided'}"
    )


def _format_recent_transcript(
    transcript: list[ConversationMessage],
    *,
    max_items: int = 8,
) -> str:
    if not transcript:
        return "No prior transcript."
    recent = transcript[-max_items:]
    lines: list[str] = []
    for item in recent:
        stage = item.stage.value if item.stage else "none"
        lines.append(f"{item.role.value}@{stage}: {item.text}")
    return "\n".join(lines)


def _build_resolution_prompt_context(payload: ResolutionVoiceCallCreateInput) -> str:
    resolution_prompt = load_stage_prompt(PipelineStage.RESOLUTION)
    handoff = ""
    if payload.completed_stages:
        handoff = build_handoff_summary(
            payload.completed_stages,
            payload.borrower.model_dump(mode="json"),
            [message.model_dump(mode="json") for message in payload.transcript],
            target_stage=PipelineStage.RESOLUTION.value,
        )

    prompt_parts = [resolution_prompt]
    prompt_parts.append("Borrower Snapshot:\n" + _borrower_snapshot(payload.borrower))
    if handoff:
        prompt_parts.append("Prior Stage Handoff (JSON):\n" + handoff)
    prompt_parts.append(
        "Recent Transcript:\n" + _format_recent_transcript(payload.transcript)
    )
    prompt_parts.append(
        "Goal: Complete the resolution conversation by presenting options, "
        "capturing borrower stance, and landing commitment or disposition."
    )
    return "\n\n".join(prompt_parts)


def _build_transient_assistant_payload(
    payload: ResolutionVoiceCallCreateInput,
) -> dict[str, Any]:
    if payload.voice_settings.mode != VoiceMode.VAPI:
        raise RuntimeError("Cannot build Vapi assistant payload when mode is not vapi.")
    if not payload.voice_settings.webhook_base_url:
        raise RuntimeError("VAPI_WEBHOOK_BASE_URL must be configured for Vapi mode.")

    server = {
        "url": f"{payload.voice_settings.webhook_base_url.rstrip('/')}/webhooks/vapi",
    }
    if payload.voice_settings.webhook_credential_id:
        server["credentialId"] = payload.voice_settings.webhook_credential_id

    workflow_metadata = {
        "workflow_id": payload.workflow_id,
        "borrower_id": payload.borrower.borrower_id,
        "stage": payload.stage.value,
    }

    assistant: dict[str, Any] = {
        "model": {
            "provider": payload.voice_settings.model_provider or _DEFAULT_MODEL_PROVIDER,
            "model": payload.voice_settings.model_name or _DEFAULT_MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": _build_resolution_prompt_context(payload),
                }
            ],
        },
        "server": server,
        "serverMessages": ["status-update", "end-of-call-report"],
        "metadata": workflow_metadata,
    }

    if payload.voice_settings.voice_provider and payload.voice_settings.voice_id:
        assistant["voice"] = {
            "provider": payload.voice_settings.voice_provider,
            "voiceId": payload.voice_settings.voice_id,
        }
    if payload.voice_settings.transcriber_provider:
        assistant["transcriber"] = {
            "provider": payload.voice_settings.transcriber_provider,
        }

    return assistant


def _build_call_create_payload(
    payload: ResolutionVoiceCallCreateInput,
) -> dict[str, Any]:
    metadata = {
        "workflow_id": payload.workflow_id,
        "borrower_id": payload.borrower.borrower_id,
        "stage": payload.stage.value,
    }
    return {
        "type": "webCall",
        "name": f"resolution-{payload.workflow_id}",
        "assistant": _build_transient_assistant_payload(payload),
        "metadata": metadata,
    }


def _extract_error_body(exc: url_error.HTTPError) -> str:
    try:
        raw = exc.read()
    except Exception:
        return ""
    try:
        return raw.decode("utf-8")
    except Exception:
        return ""


def _post_json(url: str, payload: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    req = url_request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with url_request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            if not body:
                return {}
            return json.loads(body)
    except url_error.HTTPError as exc:
        error_body = _extract_error_body(exc)
        logger.error(
            "Vapi create call failed status=%s body=%s",
            exc.code,
            error_body[:500] if error_body else "none",
        )
        raise RuntimeError(f"Vapi create call failed with status {exc.code}") from exc
    except url_error.URLError as exc:
        raise RuntimeError(f"Vapi create call network error: {exc}") from exc


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
            continue
        return value
    return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_call_start_response(
    response_payload: dict[str, Any],
) -> ResolutionVoiceCallCreateOutput:
    call_obj = _as_dict(response_payload.get("call")) or response_payload
    call_id = _first_non_empty(
        response_payload.get("id"),
        response_payload.get("callId"),
        call_obj.get("id"),
        call_obj.get("callId"),
    )
    if not call_id:
        raise RuntimeError("Vapi create call response missing call id.")

    web_call_url = _first_non_empty(
        response_payload.get("webCallUrl"),
        response_payload.get("joinUrl"),
        response_payload.get("url"),
        _as_dict(response_payload.get("webCall")).get("url"),
        call_obj.get("webCallUrl"),
        call_obj.get("joinUrl"),
        call_obj.get("url"),
    )
    call_status = _first_non_empty(
        response_payload.get("status"),
        call_obj.get("status"),
        "queued",
    )
    return ResolutionVoiceCallCreateOutput(
        provider=VoiceProvider.VAPI,
        call_id=str(call_id),
        web_call_url=str(web_call_url) if web_call_url else None,
        status=str(call_status) if call_status else None,
        metadata={
            "response_has_web_call_url": bool(web_call_url),
        },
    )


def _create_resolution_web_call_sync(
    payload: ResolutionVoiceCallCreateInput,
) -> ResolutionVoiceCallCreateOutput:
    if payload.voice_settings.mode != VoiceMode.VAPI:
        raise RuntimeError("Resolution voice call requested while mode is not vapi.")

    api_key = _env("VAPI_API_KEY")
    if not api_key:
        raise RuntimeError("VAPI_API_KEY is required for Vapi mode.")

    api_base = _env("VAPI_API_BASE_URL") or _VAPI_API_BASE_URL
    url = f"{api_base.rstrip('/')}/call"
    call_payload = _build_call_create_payload(payload)

    logger.info(
        "Creating Vapi web call workflow=%s borrower=%s stage=%s",
        payload.workflow_id,
        payload.borrower.borrower_id,
        payload.stage.value,
    )
    response_payload = _post_json(url, call_payload, api_key=api_key)
    return _normalize_call_start_response(response_payload)


async def create_resolution_web_call(
    payload: ResolutionVoiceCallCreateInput,
) -> ResolutionVoiceCallCreateOutput:
    return await asyncio.to_thread(_create_resolution_web_call_sync, payload)


def _extract_metadata(payload: dict[str, Any], call_obj: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(payload.get("metadata"))
    if metadata:
        return metadata
    metadata = _as_dict(call_obj.get("metadata"))
    if metadata:
        return metadata
    assistant = _as_dict(call_obj.get("assistant"))
    return _as_dict(assistant.get("metadata"))


def normalize_vapi_server_message(payload: dict[str, Any]) -> ResolutionVoiceEvent:
    call_obj = _as_dict(payload.get("call"))
    metadata = _extract_metadata(payload, call_obj)

    stage_raw = _first_non_empty(metadata.get("stage"), PipelineStage.RESOLUTION.value)
    try:
        stage = PipelineStage(str(stage_raw))
    except ValueError:
        stage = PipelineStage.RESOLUTION

    message_type = str(_first_non_empty(payload.get("type"), "unknown"))
    call_status = _first_non_empty(payload.get("status"), call_obj.get("status"))
    ended_reason = _first_non_empty(
        payload.get("endedReason"),
        call_obj.get("endedReason"),
    )
    call_id = _first_non_empty(
        payload.get("callId"),
        payload.get("id"),
        call_obj.get("id"),
        call_obj.get("callId"),
    )
    artifact = _as_dict(payload.get("artifact")) or _as_dict(call_obj.get("artifact"))
    messages = _as_list(payload.get("messagesOpenAIFormatted"))
    if not messages:
        messages = _as_list(payload.get("messages"))
    if not messages and artifact:
        messages = _as_list(artifact.get("messages"))

    return ResolutionVoiceEvent(
        event_type=message_type,
        workflow_id=_first_non_empty(
            metadata.get("workflow_id"),
            metadata.get("workflowId"),
        ),
        borrower_id=_first_non_empty(
            metadata.get("borrower_id"),
            metadata.get("borrowerId"),
        ),
        stage=stage,
        provider=VoiceProvider.VAPI,
        call_id=str(call_id) if call_id else None,
        call_status=str(call_status) if call_status else None,
        ended_reason=str(ended_reason) if ended_reason else None,
        final_artifact_received=message_type == "end-of-call-report",
        artifact=artifact,
        messages=messages,
        metadata={
            "source": "vapi_webhook",
            "resolved_stage": stage.value,
            "raw_type": message_type,
        },
    )
