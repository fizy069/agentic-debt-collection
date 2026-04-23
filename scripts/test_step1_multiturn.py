from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def _poll_status(base_url: str, workflow_id: str, timeout_seconds: int = 120) -> dict[str, Any]:
    start = time.time()
    status_url = f"{base_url}/pipelines/{workflow_id}"
    while True:
        status_payload = _request_json("GET", status_url)
        if status_payload.get("completed") or status_payload.get("failed"):
            return status_payload
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for workflow completion: {workflow_id}"
            )
        time.sleep(1.0)


def _poll_stage(
    base_url: str,
    workflow_id: str,
    expected_stage: str,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    start = time.time()
    status_url = f"{base_url}/pipelines/{workflow_id}"
    while True:
        status_payload = _request_json("GET", status_url)
        if status_payload.get("current_stage") == expected_stage:
            return status_payload
        if status_payload.get("failed"):
            return status_payload
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for stage '{expected_stage}' in workflow '{workflow_id}'."
            )
        time.sleep(1.0)


def _send_message(base_url: str, workflow_id: str, message: str, message_id: str) -> dict[str, Any]:
    return _request_json(
        "POST",
        f"{base_url}/pipelines/{workflow_id}/messages",
        payload={
            "message": message,
            "message_id": message_id,
        },
    )


def _seed_test_account(base_url: str) -> None:
    """Ensure a test account exists in the in-memory store.

    The account store ships with seed accounts B-TEST-001 through
    B-TEST-005.  For the scripted test we reuse B-TEST-001.
    """


def run_test(base_url: str) -> int:
    _seed_test_account(base_url)

    start_payload = {
        "borrower_id": "B-TEST-001",
        "borrower_message": "I received your notice and need clarity.",
    }
    started = _request_json("POST", f"{base_url}/pipelines", payload=start_payload)
    workflow_id = started["workflow_id"]
    print(f"Started workflow: {workflow_id}")

    _send_message(
        base_url,
        workflow_id,
        "My name is Alex Doe, date of birth is 1990-01-01, and last four is 1234.",
        "step1-assessment-1",
    )
    _send_message(
        base_url,
        workflow_id,
        "I acknowledge the balance and can pay monthly around 400 dollars.",
        "step1-assessment-2",
    )

    status = _poll_stage(base_url, workflow_id, "resolution")
    if status.get("failed"):
        print(json.dumps(status, indent=2))
        raise RuntimeError("Workflow failed before reaching resolution stage.")
    print("Reached resolution stage.")

    _send_message(
        base_url,
        workflow_id,
        "I reviewed the options and choose the payment plan.",
        "step1-resolution-1",
    )
    _send_message(
        base_url,
        workflow_id,
        "I agree to commit and can pay on schedule.",
        "step1-resolution-2",
    )

    status = _poll_stage(base_url, workflow_id, "final_notice")
    if status.get("failed"):
        print(json.dumps(status, indent=2))
        raise RuntimeError("Workflow failed before reaching final notice stage.")
    print("Reached final_notice stage.")

    _send_message(
        base_url,
        workflow_id,
        "I understand and acknowledge the final notice.",
        "step1-final-1",
    )

    final_status = _poll_status(base_url, workflow_id, timeout_seconds=120)
    print("Final status:")
    print(json.dumps(final_status, indent=2))

    if final_status.get("failed"):
        raise RuntimeError("Workflow reported failure in final status.")

    outputs = final_status.get("outputs", [])
    if not outputs:
        raise RuntimeError("Expected non-empty outputs in final status.")

    seen_stages = [item.get("stage") for item in outputs]
    for required in ("assessment", "resolution", "final_notice"):
        if required not in seen_stages:
            raise RuntimeError(f"Missing stage output for '{required}'.")

    print("Step1 multi-turn workflow test passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Step 1 multi-turn workflow smoke test."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running API.",
    )
    args = parser.parse_args()

    try:
        return run_test(args.base_url.rstrip("/"))
    except Exception as exc:
        print(f"Test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
