from __future__ import annotations

import asyncio
import base64
import logging
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from app.logging_config import setup_logging
from app.models.pipeline import (
    BorrowerMessageRequest,
    BorrowerMessageResponse,
    BorrowerRequest,
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatus,
)
from app.services.account_store import get_account_store
from app.services.voice_client import (
    MAX_UPLOAD_BYTES,
    VoiceClient,
    VoiceServiceError,
    _ALLOWED_AUDIO_MIMES,
    is_voice_enabled,
)
from app.workflows.borrower_workflow import BorrowerWorkflow


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Post default debt-collection", version="0.1.0")

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/test")
async def test_console() -> FileResponse:
    return FileResponse(_STATIC_DIR / "test.html", media_type="text/html")


@app.on_event("startup")
async def startup() -> None:
    temporal_server = os.getenv("TEMPORAL_SERVER_URL", "localhost:7233")
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "borrower-pipeline-task-queue")

    logger.info(
        "API startup  temporal=%s  namespace=%s  queue=%s",
        temporal_server, temporal_namespace, task_queue,
    )

    app.state.temporal_client = await Client.connect(
        temporal_server,
        namespace=temporal_namespace,
    )
    app.state.task_queue = task_queue
    logger.info("API connected to Temporal")


def get_temporal_client(request: Request) -> Client:
    client = getattr(request.app.state, "temporal_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal client not initialized.",
        )
    return client


def get_task_queue(request: Request) -> str:
    task_queue = getattr(request.app.state, "task_queue", None)
    if not task_queue:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task queue not configured.",
        )
    return task_queue


def _is_enabled_env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/accounts")
async def list_accounts() -> dict[str, list[str]]:
    """Return available borrower IDs from the account store."""
    store = get_account_store()
    return {"borrower_ids": store.list_ids()}


@app.post(
    "/pipelines",
    response_model=PipelineStartResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_pipeline(
    payload: PipelineStartRequest,
    client: Client = Depends(get_temporal_client),
    task_queue: str = Depends(get_task_queue),
) -> PipelineStartResponse:
    store = get_account_store()
    account = store.get(payload.borrower_id)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No account found for borrower_id '{payload.borrower_id}'.",
        )

    borrower = BorrowerRequest.from_account(account, payload.borrower_message)
    workflow_id = payload.workflow_id or f"pipeline-{borrower.borrower_id}-{uuid4().hex[:8]}"

    logger.info("POST /pipelines  workflow_id=%s  borrower=%s", workflow_id, borrower.borrower_id)

    try:
        handle = await client.start_workflow(
            BorrowerWorkflow.run,
            {"borrower": borrower.model_dump(mode="json")},
            id=workflow_id,
            task_queue=task_queue,
            execution_timeout=timedelta(hours=24),
        )
    except RPCError as exc:
        logger.error("Failed to start workflow %s: %s", workflow_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {exc}",
        ) from exc

    logger.info("Pipeline started  workflow_id=%s  run_id=%s", workflow_id, handle.first_execution_run_id)
    return PipelineStartResponse(
        workflow_id=workflow_id,
        run_id=handle.first_execution_run_id,
        task_queue=task_queue,
    )


@app.post(
    "/pipelines/{workflow_id}/messages",
    response_model=BorrowerMessageResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def send_borrower_message(
    workflow_id: str,
    payload: BorrowerMessageRequest,
    client: Client = Depends(get_temporal_client),
) -> BorrowerMessageResponse:
    handle = client.get_workflow_handle(workflow_id)

    logger.info("POST /pipelines/%s/messages  msg_id=%s", workflow_id, payload.message_id)

    try:
        await handle.signal(
            BorrowerWorkflow.add_borrower_message,
            payload.model_dump(mode="json"),
        )
    except RPCError as exc:
        logger.error("Signal failed for %s: %s", workflow_id, exc)
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send borrower message: {exc}",
        ) from exc

    logger.info("Message accepted  workflow_id=%s  msg_id=%s", workflow_id, payload.message_id)
    return BorrowerMessageResponse(
        workflow_id=workflow_id,
        accepted=True,
        message_id=payload.message_id,
    )


@app.get("/pipelines/{workflow_id}", response_model=PipelineStatus)
async def get_pipeline_status(
    workflow_id: str,
    client: Client = Depends(get_temporal_client),
) -> PipelineStatus:
    handle = client.get_workflow_handle(workflow_id)

    try:
        status_payload = await handle.query(BorrowerWorkflow.get_status)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query workflow: {exc}",
        ) from exc

    return PipelineStatus.model_validate(status_payload)


# ---------------------------------------------------------------------------
# Voice layer (Resolution stage only, env-toggled)
# ---------------------------------------------------------------------------

@app.get("/config")
async def get_config() -> dict[str, object]:
    voice_enabled = is_voice_enabled()
    show_internal_metadata = _is_enabled_env_flag("TEST_UI_SHOW_METADATA", default=False)

    cfg: dict[str, object] = {
        "agent2_voice_enabled": voice_enabled,
        "voice_enabled": voice_enabled,
        "show_internal_metadata": show_internal_metadata,
    }
    if show_internal_metadata:
        cfg["voice_stage"] = "resolution"
    return cfg


_VOICE_POLL_INTERVAL_S = 0.25
_VOICE_POLL_TIMEOUT_S = 60.0


@app.post("/pipelines/{workflow_id}/voice-turn")
async def voice_turn(
    workflow_id: str,
    audio: UploadFile,
    client: Client = Depends(get_temporal_client),
) -> dict[str, object]:
    if not is_voice_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice mode is not enabled. Set AGENT2_VOICE_ENABLED=true.",
        )

    content_type = (audio.content_type or "").split(";")[0].strip().lower()
    if content_type not in _ALLOWED_AUDIO_MIMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio type '{audio.content_type}'. Allowed: {sorted(_ALLOWED_AUDIO_MIMES)}",
        )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty audio file.",
        )
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio file too large ({len(audio_bytes)} bytes). Max {MAX_UPLOAD_BYTES}.",
        )

    handle = client.get_workflow_handle(workflow_id)

    try:
        current_status = await handle.query(BorrowerWorkflow.get_status)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query workflow: {exc}",
        ) from exc

    if current_status.get("current_stage") != "resolution":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Voice turns are only accepted during the resolution stage. Current stage: {current_status.get('current_stage')}",
        )

    pre_transcript_len = len(current_status.get("transcript", []))

    try:
        vc = VoiceClient()
    except VoiceServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Voice service initialization failed: {exc.detail}",
        ) from exc

    safe_filename = f"voice-{uuid4().hex[:8]}.webm"
    try:
        borrower_text = await vc.transcribe(audio_bytes, safe_filename)
    except VoiceServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"STT failed: {exc.detail}",
        ) from exc

    if not borrower_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not transcribe any speech from the audio.",
        )

    logger.info(
        "voice_turn  workflow_id=%s  stt_text_len=%d",
        workflow_id, len(borrower_text),
    )

    message_id = f"voice-{uuid4().hex[:8]}"
    try:
        await handle.signal(
            BorrowerWorkflow.add_borrower_message,
            {"message": borrower_text, "message_id": message_id},
        )
    except RPCError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to signal workflow: {exc}",
        ) from exc

    assistant_reply = None
    poll_status = None
    elapsed = 0.0
    while elapsed < _VOICE_POLL_TIMEOUT_S:
        await asyncio.sleep(_VOICE_POLL_INTERVAL_S)
        elapsed += _VOICE_POLL_INTERVAL_S
        try:
            poll_status = await handle.query(BorrowerWorkflow.get_status)
        except RPCError:
            continue

        transcript = poll_status.get("transcript", [])
        if len(transcript) >= pre_transcript_len + 2:
            last_msg = transcript[-1]
            if last_msg.get("role") == "agent":
                assistant_reply = last_msg.get("text", "")
                break

    if assistant_reply is None:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Timed out waiting for the agent to respond.",
        )

    try:
        tts_bytes, tts_mime = await vc.synthesize(assistant_reply)
    except VoiceServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"TTS failed: {exc.detail}",
        ) from exc

    audio_b64 = base64.b64encode(tts_bytes).decode("ascii")

    return {
        "transcribed_text": borrower_text,
        "assistant_reply": assistant_reply,
        "audio_base64": audio_b64,
        "audio_mime": tts_mime,
        "current_stage": poll_status.get("current_stage", "resolution") if poll_status else "resolution",
        "stage_complete": poll_status.get("completed", False) if poll_status else False,
    }


@app.get("/voice-greeting")
async def voice_greeting(
    workflow_id: str = Query(...),
    client: Client = Depends(get_temporal_client),
) -> dict[str, object]:
    """TTS the first resolution-stage agent reply so the call opens with a spoken greeting."""
    if not is_voice_enabled():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Voice mode is not enabled.",
        )

    handle = client.get_workflow_handle(workflow_id)
    try:
        wf_status = await handle.query(BorrowerWorkflow.get_status)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow '{workflow_id}' not found.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query workflow: {exc}",
        ) from exc

    transcript = wf_status.get("transcript", [])
    greeting_text = None
    for msg in reversed(transcript):
        if msg.get("role") == "agent" and msg.get("stage") == "resolution":
            greeting_text = msg.get("text", "")
            break

    if not greeting_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No resolution agent reply found in transcript yet.",
        )

    try:
        vc = VoiceClient()
        audio_bytes, mime = await vc.synthesize(greeting_text)
    except VoiceServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Greeting TTS failed: {exc.detail}",
        ) from exc

    return {
        "assistant_reply": greeting_text,
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "audio_mime": mime,
    }
