"""CLI to roll back a prompt section to a prior version.

Usage::

    python -m app.eval.rollback --section "system_template:assessment" --version v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

from app.eval.audit_trail import AuditTrail
from app.services.prompt_registry import get_prompt_registry, reset_prompt_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll back a prompt section to a specific historical version.",
    )
    parser.add_argument(
        "--section", required=True,
        help="Section key to roll back (e.g. 'system_template:assessment').",
    )
    parser.add_argument(
        "--version", required=True,
        help="Target version to restore (e.g. 'v1').",
    )
    parser.add_argument(
        "--audit-dir", default="data/audit",
        help="Audit trail base directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = AuditTrail(base_dir=args.audit_dir)

    reset_prompt_registry()
    registry = get_prompt_registry(audit_trail=audit)

    available = registry.list_versions()
    section_versions = available.get(args.section)
    if section_versions is None:
        print(f"ERROR: Section '{args.section}' not found.", file=sys.stderr)
        print(f"Available sections: {list(available.keys())}", file=sys.stderr)
        sys.exit(1)

    if args.version not in section_versions:
        print(
            f"ERROR: Version '{args.version}' not in history for '{args.section}'.",
            file=sys.stderr,
        )
        print(f"Available versions: {section_versions}", file=sys.stderr)
        sys.exit(1)

    restored = registry.rollback(args.section, args.version)
    result = {
        "section_key": args.section,
        "restored_version": restored.version,
        "content_length": len(restored.content),
        "content_preview": restored.content[:200],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
