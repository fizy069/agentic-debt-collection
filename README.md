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
