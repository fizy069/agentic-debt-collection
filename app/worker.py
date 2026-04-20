from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from app.activities.agents import (
    assessment_agent,
    final_notice_agent,
    resolution_agent,
)
from app.logging_config import setup_logging
from app.workflows.borrower_workflow import BorrowerWorkflow

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    setup_logging()

    temporal_server = os.getenv("TEMPORAL_SERVER_URL", "localhost:7233")
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "borrower-pipeline-task-queue")

    logger.info(
        "Worker starting  server=%s  namespace=%s  queue=%s",
        temporal_server, temporal_namespace, task_queue,
    )

    client = await Client.connect(temporal_server, namespace=temporal_namespace)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BorrowerWorkflow],
        activities=[assessment_agent, resolution_agent, final_notice_agent],
    )
    logger.info("Worker connected and running")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
