"""Manual smoke test: upsert violations into the vector DB, then query them back.

Run from the repo root:
    python tests/manual_vector_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.compliance_vector_store import (
    _get_collection,
    get_client,
    query_similar_violations,
    reset_client,
    upsert_violations,
)


def _separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    reset_client()

    _separator("1. Initialize client & show DB path")
    client = get_client()
    col = _get_collection()
    print(f"  Collection : {col.name}")
    print(f"  Count before: {col.count()}")

    _separator("2. Upsert sample violations")
    sample_violations = [
        {
            "rule": "2",
            "label": "false_threat",
            "confidence": 0.95,
            "excerpt": "you will be arrested if you don't pay",
        },
        {
            "rule": "4",
            "label": "offer_out_of_bounds",
            "confidence": 0.88,
            "excerpt": "we can settle for 10% of the balance",
        },
        {
            "rule": "8",
            "label": "pii_leak",
            "confidence": 0.99,
            "excerpt": "your SSN 123-45-6789 is on file",
        },
        {
            "rule": "7",
            "label": "unprofessional_tone",
            "confidence": 0.72,
            "excerpt": "you're being ridiculous about this debt",
        },
    ]

    n = upsert_violations(
        workflow_id="manual-test-wf",
        stage="resolution",
        turn_index=1,
        violations=sample_violations,
    )
    print(f"  Upserted {n} records")
    print(f"  Count after : {col.count()}")

    _separator("3. Query: 'arrested for not paying' (expect false_threat hit)")
    result = query_similar_violations("arrested for not paying", n_results=3)
    if result.error:
        print(f"  ERROR: {result.error}")
    elif not result.records:
        print("  No results found")
    else:
        for r in result.records:
            print(f"  [{r.rule}] {r.label}  (conf={r.confidence:.2f})  excerpt: {r.excerpt}")

    _separator("4. Query: 'SSN leak' (expect pii_leak hit)")
    result = query_similar_violations("SSN shown in message", n_results=3)
    if result.error:
        print(f"  ERROR: {result.error}")
    else:
        for r in result.records:
            print(f"  [{r.rule}] {r.label}  (conf={r.confidence:.2f})  excerpt: {r.excerpt}")

    _separator("5. Query: 'settle for 5%' (expect offer_out_of_bounds hit)")
    result = query_similar_violations("we can settle for 5% of the total", n_results=3)
    if result.error:
        print(f"  ERROR: {result.error}")
    else:
        for r in result.records:
            print(f"  [{r.rule}] {r.label}  (conf={r.confidence:.2f})  excerpt: {r.excerpt}")

    _separator("6. Query with stage filter: stage='assessment' (should find nothing)")
    result = query_similar_violations("arrested", stage="assessment", n_results=3)
    print(f"  Records returned: {len(result.records)}")
    if result.records:
        for r in result.records:
            print(f"  [{r.rule}] {r.label} stage={r.stage}")

    _separator("7. Upsert a second workflow, then query across all")
    upsert_violations(
        workflow_id="manual-test-wf-2",
        stage="assessment",
        turn_index=2,
        violations=[
            {
                "rule": "5",
                "label": "hardship_dismissed",
                "confidence": 0.80,
                "excerpt": "that's not our problem, pay up",
            },
        ],
    )
    result = query_similar_violations("borrower hardship ignored", n_results=5)
    print(f"  Total hits: {len(result.records)}")
    for r in result.records:
        print(f"  [{r.rule}] {r.label}  stage={r.stage}  turn={r.turn_index}  excerpt: {r.excerpt}")

    _separator("8. Raw collection peek (all IDs + metadatas)")
    peek = col.get(limit=20, include=["metadatas", "documents"])
    for i, vid in enumerate(peek["ids"]):
        meta = peek["metadatas"][i] if peek["metadatas"] else {}
        doc = (peek["documents"][i] if peek["documents"] else "")[:80]
        print(f"  {vid}  rule={meta.get('rule')}  label={meta.get('label')}  doc={doc}...")

    _separator("DONE")
    print(f"  Final collection count: {col.count()}")
    print("  DB is persistent — re-run this script and count will include old data.")
    print("  Delete data/compliance_vectors/ to start fresh.\n")


if __name__ == "__main__":
    main()
