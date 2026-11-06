"""Deterministic extractive answers for temporary LLM provider failures."""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.rag.context import ContextPackage, Citation


_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "which",
        "who",
        "with",
    }
)


@dataclass(frozen=True)
class _ScoredSentence:
    text: str
    candidate_index: int
    score: float


def _terms(text: str) -> set[str]:
    return {
        token
        for token in _WORD_PATTERN.findall(text.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _sentence_score(sentence: str, query_terms: set[str], query: str) -> float:
    sentence_terms = _terms(sentence)
    overlap = sentence_terms & query_terms
    if not overlap:
        return 0.0
    score = float(len(overlap) * 3)
    if any(char.isdigit() for char in sentence):
        score += 1.5
    if "%" in sentence or re.search(r"\b(?:accuracy|rate|score|metric|value|total|average)\b", sentence, re.I):
        score += 1.0
    if any(char.isdigit() for char in query) and any(char.isdigit() for char in sentence):
        score += 2.0
    return score


def _split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_PATTERN.split(text.strip()) if sentence.strip()]


def build_extractive_fallback(
    query: str, context: ContextPackage
) -> tuple[str, list[Citation], bool]:
    """Return an evidence-only answer, its source citations, and insufficiency."""
    query_terms = _terms(query)
    scored: list[_ScoredSentence] = []
    for candidate_index, candidate in enumerate(context.candidates):
        for sentence in _split_sentences(candidate.text):
            score = _sentence_score(sentence, query_terms, query)
            if score > 0:
                retrieval_score = (
                    candidate.cross_encoder_score
                    or candidate.rrf_score
                    or candidate.bm25_score
                    or candidate.dense_score
                    or 0.0
                )
                scored.append(
                    _ScoredSentence(
                        sentence,
                        candidate_index,
                        score + min(max(float(retrieval_score), 0.0), 1.0),
                    )
                )
    if not scored:
        return (
            "The indexed documents do not contain enough information to establish an answer.",
            [],
            True,
        )

    selected: list[_ScoredSentence] = []
    seen_sentences: set[str] = set()
    for item in sorted(scored, key=lambda value: (-value.score, value.candidate_index)):
        normalized = item.text.casefold()
        if normalized in seen_sentences:
            continue
        selected.append(item)
        seen_sentences.add(normalized)
        if len(selected) == 3:
            break

    citation_indexes = {item.candidate_index for item in selected}
    citations = [
        citation for index, citation in enumerate(context.citations) if index in citation_indexes
    ]
    evidence = " ".join(item.text for item in selected)
    return f"Based on the indexed documents: {evidence}", citations, False
