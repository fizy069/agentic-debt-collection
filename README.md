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

### 5) Run a pipeline

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

Expected behavior:
- workflow stages move in strict order:
  `assessment -> resolution -> final_notice -> completed`
- `GET /pipelines/{id}` returns stage outputs and final outcome.
