from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from temporalio.client import Client
from temporalio.worker import Worker

from app.activities.agents import (
    assessment_agent,
    final_notice_agent,
    resolution_agent,
)
from app.workflows.borrower_workflow import BorrowerWorkflow


load_dotenv()


async def run_worker() -> None:
    temporal_server = os.getenv("TEMPORAL_SERVER_URL", "localhost:7233")
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "borrower-pipeline-task-queue")

    client = await Client.connect(temporal_server, namespace=temporal_namespace)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BorrowerWorkflow],
        activities=[assessment_agent, resolution_agent, final_notice_agent],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(run_worker())
