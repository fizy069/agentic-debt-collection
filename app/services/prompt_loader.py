from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.models.pipeline import PipelineStage

_PROMPT_FILES = {
    PipelineStage.ASSESSMENT: "assessment.txt",
    PipelineStage.RESOLUTION: "resolution.txt",
    PipelineStage.FINAL_NOTICE: "final_notice.txt",
}


@lru_cache(maxsize=3)
def load_stage_prompt(stage: PipelineStage) -> str:
    """Load stage prompt text from ``app/prompts/<stage>.txt``."""
    filename = _PROMPT_FILES[stage]
    path = Path(__file__).resolve().parent.parent / "prompts" / filename
    return path.read_text(encoding="utf-8").strip()
