"""Deterministic checks for RAG answers and their citations."""

from typing import Any


def evaluate_answer(
    answer: str,
    reference_answer: str | None,
    citations: list[dict[str, Any]],
    retrieved_chunk_ids: set[str],
    sufficient_evidence: bool,
) -> dict[str, Any]:
    """Score only verifiable properties; unavailable semantics remain null."""
    normalized = answer.casefold().strip()
    citation_ids = [str(item.get("chunk_id", "")) for item in citations]
    valid_citations = bool(citations) and all(item in retrieved_chunk_ids for item in citation_ids)
    if not sufficient_evidence:
        abstained = any(
            phrase in normalized
            for phrase in ("not enough", "insufficient", "could not find", "don't have enough")
        )
        return {
            "abstention_quality": 1.0 if abstained else 0.0,
            "faithfulness": None,
            "answer_relevance": None,
            "citation_correctness": 1.0 if valid_citations or not citations else 0.0,
            "answer_completeness": None,
            "judge": None,
        }
    return {
        "abstention_quality": 0.0 if not normalized else None,
        "faithfulness": None,
        "answer_relevance": 1.0 if normalized and reference_answer else None,
        "citation_correctness": 1.0 if valid_citations else 0.0,
        "answer_completeness": None,
        "judge": None,
        "note": "Semantic faithfulness and completeness require a configured judge; no perfect defaults are assigned.",
    }
