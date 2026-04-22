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

### 8) Agent 2 voice layer (optional)

The Resolution stage can use voice instead of text. The borrower speaks into the
browser microphone, audio is transcribed via OpenAI STT, the existing workflow
processes the turn as usual, and the agent reply is spoken back via OpenAI TTS.

Enable it by setting these in `.env`:

```
OPENAI_API_KEY=sk-...
AGENT2_VOICE_ENABLED=true
```

Optional tuning (defaults shown):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_STT_MODEL` | `gpt-4o-mini-transcribe` | OpenAI transcription model |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | OpenAI text-to-speech model |
| `OPENAI_TTS_VOICE` | `alloy` | TTS voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |

Voice is independent of the LLM provider: agents can use Anthropic while STT/TTS
use OpenAI.

With voice enabled, open `http://127.0.0.1:8000/test` and progress through
Assessment by typing. When Resolution starts, an "Incoming Resolution Call" card
appears. Click **Accept** to switch to the voice call panel. The agent speaks a
greeting, then the microphone activates with automatic voice activity detection
(VAD) — no click-to-talk needed. When you stop speaking, a filler phrase plays
immediately while the system processes your message (STT → agent → TTS). The
agent's reply then plays automatically and the mic reopens. Click **Decline** to
stay in text mode. The call automatically ends when Resolution completes and the
pipeline returns to text for Final Notice.

Note: browser microphone access requires a secure context. `localhost` and
`127.0.0.1` work for local development; production deployments need HTTPS.
