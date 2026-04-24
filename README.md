# What is this repo about?

This repo contains a 3-agent chat/voice workflow that acts as a post-default debt collection system. 

Some key architectural decisions taken: 
1. Since we have a tight 2000 token budget for the entire agent, we need to scope each agent tightly. Handoffs and summarization must be deterministic. For this to be possible, i've templated handoff and summarization into fixed JSON schemas. This ensures token efficient info transfer without loss of crucial info. 

2. Compliance : For compliance, we have a 2 layer system. Layer 1 performs rudimentary checks, and also performs a vector DB lookup for possible violations. Layer 2 is an LLM-as-a-judge layer that populates the layer 1 database.
   As more users register interactions with the agent, we have fewer cache misses, and eventually the higher-token costing LLM as a judge API call is invoked lesser.

3. Self learning : Treating the agent's prompts, and the judges prompts as first-class citizens allows for cleanly implementing a self learning loop, and a meta evaluation layer that evaluates the judges themselves, hence satisfying the Darwin Godel Model design for the system.






# Debt Resolution Pipeline

Temporal-orchestrated borrower workflow for a post-default debt collection flow. The project combines a FastAPI API, a Temporal worker, stage-specific agent activities, compliance guardrails, token-budgeted handoffs, and an optional browser-based voice experience for the Resolution stage.

The current runtime implements the core borrower pipeline from the assignment: `assessment -> resolution -> final_notice`. The repository also includes design and planning documents for the broader self-learning system, but the code in this repo is centered on the live workflow, compliance, and voice path.

## What the project does

- Runs one Temporal workflow per borrower.
- Supports multi-turn conversations inside each stage before advancing.
- Preserves cross-stage continuity with deterministic handoff summaries capped at 500 tokens.
- Enforces a 2000-token agent context budget with overflow summarization safeguards.
- Applies deterministic compliance controls, an optional LLM compliance judge, and a local Chroma-backed violation memory.
- Supports Anthropic-first LLM execution, OpenAI fallback, and stub mode when no live key is configured.
- Optionally switches the Resolution stage from chat to voice using OpenAI STT/TTS.

## Runtime layout

- `app/main.py`: FastAPI API for workflow start, borrower messaging, status queries, voice endpoints, and static assets.
- `app/worker.py`: Temporal worker that runs the workflow and stage activities.
- `app/workflows/borrower_workflow.py`: borrower workflow state machine and stage orchestration.
- `app/activities/agents.py`: stage-turn execution, prompt assembly, handoffs, compliance checks, and LLM calls.
- `app/prompts/`: text prompts for `assessment`, `resolution`, and `final_notice`.
- `app/services/`: LLM provider facade, token budget, summarization, compliance, vector store, and voice client.
- `tests/`: unit and integration-style coverage for compliance, handoffs, summarization, token budgets, and workflow behavior.

For the more detailed system breakdown, see [architecture.md](architecture.md).

## Stage behavior

| Stage | Channel | Purpose | Completion rule |
| --- | --- | --- | --- |
| `assessment` | chat | Confirm identity, establish debt context, understand ability to pay | Required signals collected or 3 turns reached |
| `resolution` | chat or voice | Present policy-bounded options and capture borrower position | Required signals collected or 3 turns reached |
| `final_notice` | chat | Deliver final notice and record acknowledgement/response | Required signals collected or 2 turns reached |

The workflow always advances in strict order and seeds the first borrower turn from `borrower.borrower_message` in the start request.

## Prerequisites

- Python 3.11+
- Temporal CLI
- An Anthropic API key for the default live model path, or an OpenAI API key for the fallback live model path
- An OpenAI API key if you want to use Resolution-stage voice

Install Temporal CLI on Windows:

```powershell
winget install Temporal.TemporalCLI
```

If you prefer manual installation, use the Temporal CLI releases page: https://github.com/temporalio/cli/releases

## Quick start

### 1) Create a virtual environment and install the project

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
Copy-Item .env.example .env
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
```

### 2) Configure `.env`

Start from `.env.example` and update only the values you need. Provider selection is:

1. `ANTHROPIC_API_KEY` if present
2. `OPENAI_API_KEY` if no Anthropic key is present
3. stub mode if neither key is set

Stub mode is useful for local wiring and many tests, but it does not produce real model behavior. Voice always requires `OPENAI_API_KEY` even if your text agents use Anthropic.

### 3) Start the stack

Recommended launcher on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-launch.ps1
```

Recommended launcher on macOS/Linux:

```bash
bash ./scripts/dev-launch.sh
```

Useful launcher flags:

- PowerShell: `-SkipInstall`, `-OpenTerminalUI`
- Bash: `--skip-install`, `--open-ui`

#### Run with Docker

```bash
cp .env.example .env
# then edit .env with your ANTHROPIC_API_KEY / OPENAI_API_KEY
docker compose up --build
```

- API docs: `http://localhost:8000/docs`
- Test console: `http://localhost:8000/test`
- Temporal UI: `http://localhost:8233`

Manual startup is also supported.

Temporal:

```powershell
temporal server start-dev
```

Worker:

```powershell
.venv\Scripts\Activate.ps1
python -m app.worker
```

API:

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 4) Verify the services

Once the stack is running:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`
- Test console: `http://127.0.0.1:8000/test`
- Temporal UI: `http://localhost:8233`

## Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | unset | Enables Anthropic as the primary live LLM provider |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Anthropic model used for main agent turns |
| `OPENAI_API_KEY` | unset | OpenAI fallback LLM key and required key for voice |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model when Anthropic is not configured |
| `TEMPORAL_SERVER_URL` | `localhost:7233` | Temporal endpoint for the API and worker |
| `TEMPORAL_NAMESPACE` | `default` | Temporal namespace |
| `TEMPORAL_TASK_QUEUE` | `borrower-pipeline-task-queue` | Shared task queue for workflow and activities |
| `COMPLIANCE_JUDGE_ENABLED` | `false` | Enables audit-only LLM-as-a-judge checks |
| `COMPLIANCE_JUDGE_MODEL` | unset | Model used by the compliance judge when enabled |
| `COMPLIANCE_VECTOR_DB_PATH` | `data/compliance_vectors` | Persistent Chroma path for normalized violation memory |
| `AGENT2_VOICE_ENABLED` | `false` | Enables voice mode for the Resolution stage |
| `OPENAI_STT_MODEL` | `gpt-4o-mini-transcribe` | Speech-to-text model for voice turns |
| `OPENAI_TTS_MODEL` | `gpt-4o-mini-tts` | Text-to-speech model for voice turns |
| `OPENAI_TTS_VOICE` | `alloy` | Voice preset used for TTS responses |

## API quick start

### Start a workflow

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

$start = Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/pipelines" `
  -ContentType "application/json" `
  -Body $body

$start
```

`POST /pipelines` starts a new Temporal workflow and automatically seeds the initial borrower message from the request payload.

### Send a borrower message

```powershell
Invoke-RestMethod -Method Post `
  -Uri ("http://127.0.0.1:8000/pipelines/" + $start.workflow_id + "/messages") `
  -ContentType "application/json" `
  -Body (@{
    message = "My name is Alex Doe, DOB 1990-01-01, last four 1234."
    message_id = "msg-1"
  } | ConvertTo-Json)
```

`message_id` is optional but useful as an idempotency token.

### Check workflow status

```powershell
Invoke-RestMethod -Method Get `
  -Uri ("http://127.0.0.1:8000/pipelines/" + $start.workflow_id)
```

The status payload includes:

- `current_stage`, `completed`, and `failed`
- `outputs` for each executed stage turn
- full `transcript`
- `latest_assistant_reply`
- `stage_turn_counts` and `stage_collected_fields`
- `compliance_flags` and `final_outcome`

Main API endpoints:

- `POST /pipelines`
- `POST /pipelines/{workflow_id}/messages`
- `GET /pipelines/{workflow_id}`
- `GET /health`
- `GET /config`
- `GET /test`

## Voice mode for Resolution

Voice mode is optional and only applies during the `resolution` stage.

Enable it in `.env`:

```env
OPENAI_API_KEY=sk-...
AGENT2_VOICE_ENABLED=true
```

With voice enabled:

- The borrower can use the browser-based test console at `http://127.0.0.1:8000/test`.
- Assessment remains text-driven.
- When Resolution starts, the UI offers an incoming call flow.
- Borrower audio is transcribed with OpenAI STT, signaled into the existing workflow, and the reply is synthesized with OpenAI TTS.
- A pre-generated filler clip at `/static/filler.mp3` is used to fill silence while STT, workflow execution, and TTS complete.

Voice-specific API endpoints:

- `POST /pipelines/{workflow_id}/voice-turn`: accepts multipart audio uploads in `webm`, `ogg`, `mp3`, `wav`, `m4a`, or related MIME variants, up to 20 MB
- `GET /voice-greeting?workflow_id=...`: generates spoken audio for the first Resolution-stage agent reply

Local browser microphone access works on `localhost` and `127.0.0.1`. Non-local deployments need HTTPS.

## Testing

The repository includes pytest-based coverage for compliance, token budgeting, handoffs, summarization, prompt behavior, and workflow edge cases.

If you do not already have pytest installed in your environment, add the test tools first:

```powershell
python -m pip install pytest pytest-asyncio
```

Run the test suite:

```powershell
python -m pytest
```

For an end-to-end smoke test against a running Temporal stack and API:

```powershell
python .\scripts\test_step1_multiturn.py --base-url http://127.0.0.1:8000
```

That script starts a workflow, pushes borrower messages through all three stages, waits for stage transitions, and validates that the pipeline completes with outputs for `assessment`, `resolution`, and `final_notice`.

## Related documents

- [architecture.md](architecture.md): detailed system architecture and API surface
- [problem_statement.md](problem_statement.md): concise restatement of the assignment requirements
- [iterative_plan.md](iterative_plan.md): implementation plan summary
- [iterative_plan_detailed.md](iterative_plan_detailed.md): deeper execution plan
- [self-learning.md](self-learning.md): notes on the future self-learning loop
- [human_decision_journal.md](human_decision_journal.md): key trade-offs and decision log
