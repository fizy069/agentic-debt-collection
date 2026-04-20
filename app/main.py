from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from app.logging_config import setup_logging
from app.models.pipeline import (
    BorrowerMessageRequest,
    BorrowerMessageResponse,
    PipelineStartRequest,
    PipelineStartResponse,
    PipelineStatus,
)
from app.workflows.borrower_workflow import BorrowerWorkflow


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Post default debt-collection", version="0.1.0")

_STATIC_DIR = Path(__file__).resolve().parent / "static"


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


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


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
    workflow_id = payload.workflow_id or f"pipeline-{payload.borrower.borrower_id}-{uuid4().hex[:8]}"

    logger.info("POST /pipelines  workflow_id=%s  borrower=%s", workflow_id, payload.borrower.borrower_id)

    try:
        handle = await client.start_workflow(
            BorrowerWorkflow.run,
            {"borrower": payload.borrower.model_dump(mode="json")},
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
