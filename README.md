# Simple Agent Pipeline

Minimal FastAPI + Temporal scaffold for a three-stage debt collection flow.

## Services

- API: starts and inspects borrower workflows.
- Worker: runs Temporal workflow and activities.
- Temporal: workflow backend.

## Local Python Setup (No Docker)

### 1) Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env` if you want real model responses.
If you leave it unset, the activities still run with a safe stub response.

### 2) Temporal setup (required)

Install Temporal CLI on Windows:

```powershell
winget install Temporal.TemporalCLI
```

If you prefer manual install, use the official releases:
[https://github.com/temporalio/cli/releases](https://github.com/temporalio/cli/releases)

Then start a local dev server in terminal 1:

```powershell
temporal server start-dev
```

### 3) One-command dev launcher (recommended)

This starts Temporal, worker, and API in separate PowerShell windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-launch.ps1
```

The launcher now uses `.venv\Scripts\python.exe` and bootstraps dependencies automatically.
If you already installed deps and want a faster start, skip install:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-launch.ps1 -SkipInstall
```

Optional: open Temporal UI in browser immediately:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-launch.ps1 -OpenTerminalUI
```

### 4) Start worker and API manually

Terminal 2 (worker):

```powershell
.venv\Scripts\Activate.ps1
python -m app.worker
```

Terminal 3 (API):

```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 5) Run a pipeline (start workflow)

```powershell
$body = @{
  borrower = @{
    borrower_id = "b001"
    account_reference = "acct-7788"
    debt_amount = 1500
    currency = "USD"
    days_past_due = 45
    borrower_message = "I need options."
    notes = "Prefers text communication."
  }
} | ConvertTo-Json -Depth 6

$start = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/pipelines" -ContentType "application/json" -Body $body
$start

Invoke-RestMethod -Method Get -Uri ("http://127.0.0.1:8000/pipelines/" + $start.workflow_id)
```

### 6) Multi-turn borrower chat progression

Send borrower messages into the active workflow:

```powershell
Invoke-RestMethod -Method Post `
  -Uri ("http://127.0.0.1:8000/pipelines/" + $start.workflow_id + "/messages") `
  -ContentType "application/json" `
  -Body (@{ message = "My name is Alex Doe, DOB 1990-01-01, last four 1234." ; message_id = "msg-1" } | ConvertTo-Json)
```

Poll status:

```powershell
Invoke-RestMethod -Method Get -Uri ("http://127.0.0.1:8000/pipelines/" + $start.workflow_id)
```

Expected behavior:
- workflow stages progress in strict order:
  `assessment -> resolution -> final_notice -> completed`
- each stage can process multiple borrower turns before transition
- status includes transcript, latest assistant reply, per-stage turn counters, and outputs.

### 7) Scripted Step 1 smoke test

With Temporal + worker + API running:

```powershell
python .\scripts\test_step1_multiturn.py --base-url http://127.0.0.1:8000
```

The script starts a workflow, sends staged borrower messages, waits for stage transitions, and validates end-to-end completion.

## Resolution Voice Mode (Vapi)

Agent 2 (`resolution`) now supports two modes controlled by env:

- `AGENT2_VOICE_MODE=stub` (default): keeps the current text/stub Resolution path.
- `AGENT2_VOICE_MODE=vapi`: creates a transient Vapi `webCall` at Resolution stage entry and waits for webhook completion before finalizing Resolution.

Minimum env required for `vapi` mode:

```powershell
AGENT2_VOICE_MODE=vapi
VAPI_API_KEY=...
VAPI_WEBHOOK_BASE_URL=https://<public-base-url>
```

Optional env:

- `VAPI_WEBHOOK_CREDENTIAL_ID` for dashboard-managed webhook credentials
- `VAPI_MODEL_PROVIDER`, `VAPI_MODEL_NAME`, `VAPI_VOICE_PROVIDER`, `VAPI_VOICE_ID`, `VAPI_TRANSCRIBER_PROVIDER`
- `VAPI_WEBHOOK_AUTH_BEARER` and/or `VAPI_WEBHOOK_SECRET` for local validation

### Webhook endpoint

Vapi server events are received at:

```text
POST /webhooks/vapi
```

The API extracts workflow routing metadata (`workflow_id`, `borrower_id`, `stage`) from webhook payload metadata and signals the matching Temporal workflow.

### Local webhook routing notes

For local testing, the API must be reachable from Vapi over a public URL. Use a tunnel (for example `ngrok`) and point `VAPI_WEBHOOK_BASE_URL` to that public base URL:

```powershell
ngrok http 8000
```

If you run the API on a different port (for example `4242`), tunnel that port instead.

`vapi listen` is only a local forwarder. It does not replace the need for a publicly reachable URL for Vapi-originated webhooks in end-to-end testing.

### Manual browser verification

1. Open `http://127.0.0.1:8000/test`.
2. Start a pipeline and progress through Assessment.
3. At Resolution (in `vapi` mode), use the **Join Resolution Call** action when `webCallUrl` is present.
4. End the voice call and confirm the pipeline advances after webhook completion.
5. Continue Final Notice on the existing text path.
