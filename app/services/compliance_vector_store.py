"""Local persistent vector store for compliance violations.

Uses ChromaDB with a repo-local storage path (configurable via
``COMPLIANCE_VECTOR_DB_PATH``, default ``data/compliance_vectors``).

Only *redacted* text plus normalized rule labels and metadata are stored —
never raw sensitive content.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/compliance_vectors"
_COLLECTION_NAME = "compliance_violations"

_client_instance: chromadb.ClientAPI | None = None


def _get_db_path() -> str:
    configured = os.getenv("COMPLIANCE_VECTOR_DB_PATH", "").strip()
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[2] / _DEFAULT_DB_PATH)


def get_client() -> chromadb.ClientAPI:
    global _client_instance
    if _client_instance is not None:
        return _client_instance

    db_path = _get_db_path()
    Path(db_path).mkdir(parents=True, exist_ok=True)
    _client_instance = chromadb.PersistentClient(path=db_path)
    logger.info("compliance_vector_store_init  path=%s", db_path)
    return _client_instance


def reset_client() -> None:
    """Drop the cached singleton (useful in tests)."""
    global _client_instance
    _client_instance = None


def _get_collection() -> chromadb.Collection:
    client = get_client()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@dataclass(frozen=True)
class ViolationRecord:
    violation_id: str
    rule: str
    label: str
    confidence: float
    excerpt: str
    stage: str
    turn_index: int


@dataclass(frozen=True)
class VectorSearchResult:
    records: list[ViolationRecord] = field(default_factory=list)
    error: str | None = None


def upsert_violations(
    *,
    workflow_id: str,
    stage: str,
    turn_index: int,
    violations: list[dict[str, Any]],
) -> int:
    """Persist judge-classified violations into the vector store.

    Each violation is stored with its redacted excerpt as the document text
    for embedding, plus structured metadata for filtering.

    Returns the number of records upserted.
    """
    if not violations:
        return 0

    collection = _get_collection()

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for idx, v in enumerate(violations):
        vid = f"{workflow_id}_{stage}_{turn_index}_{idx}"
        doc_text = (
            f"rule:{v.get('rule', '')} label:{v.get('label', '')} "
            f"stage:{stage} excerpt:{v.get('excerpt', '')}"
        )
        ids.append(vid)
        documents.append(doc_text)
        metadatas.append(
            {
                "workflow_id": workflow_id,
                "stage": stage,
                "turn_index": turn_index,
                "rule": str(v.get("rule", "")),
                "label": str(v.get("label", "")),
                "confidence": float(v.get("confidence", 0.0)),
            },
        )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        "vector_upsert  workflow=%s  stage=%s  turn=%d  count=%d",
        workflow_id, stage, turn_index, len(ids),
    )
    return len(ids)


def query_similar_violations(
    text: str,
    *,
    n_results: int = 3,
    stage: str | None = None,
) -> VectorSearchResult:
    """Search the vector store for historically similar violations.

    Optionally filter by ``stage``.  Returns up to ``n_results`` records.
    Swallows errors so the pipeline is never blocked.
    """
    try:
        collection = _get_collection()
        if collection.count() == 0:
            return VectorSearchResult()

        where_filter: dict[str, str] | None = None
        if stage:
            where_filter = {"stage": stage}

        results = collection.query(
            query_texts=[text],
            n_results=min(n_results, collection.count()),
            where=where_filter if where_filter else None,
        )

        records: list[ViolationRecord] = []
        if results and results.get("ids"):
            ids_batch = results["ids"][0]
            metas_batch = (results.get("metadatas") or [[]])[0]
            docs_batch = (results.get("documents") or [[]])[0]

            for i, vid in enumerate(ids_batch):
                meta = metas_batch[i] if i < len(metas_batch) else {}
                doc = docs_batch[i] if i < len(docs_batch) else ""
                excerpt = doc.split("excerpt:")[-1].strip() if "excerpt:" in doc else doc
                records.append(
                    ViolationRecord(
                        violation_id=vid,
                        rule=str(meta.get("rule", "")),
                        label=str(meta.get("label", "")),
                        confidence=float(meta.get("confidence", 0.0)),
                        excerpt=excerpt,
                        stage=str(meta.get("stage", "")),
                        turn_index=int(meta.get("turn_index", 0)),
                    )
                )

        return VectorSearchResult(records=records)

    except Exception as exc:
        logger.exception("vector_query_failed")
        return VectorSearchResult(error=f"vector_query_failed: {exc}")


def vector_hits_to_metadata(result: VectorSearchResult) -> dict[str, Any]:
    """Flatten vector search results for inclusion in StageTurnOutput.metadata."""
    if result.error:
        return {"vector_lookup_error": result.error}
    if not result.records:
        return {}
    return {
        "vector_similar_violations": [
            {
                "violation_id": r.violation_id,
                "rule": r.rule,
                "label": r.label,
                "confidence": r.confidence,
                "stage": r.stage,
            }
            for r in result.records
        ]
    }
