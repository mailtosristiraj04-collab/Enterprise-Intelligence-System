# Evaluation and benchmarking

## Dataset format

Evaluation data is JSONL and is separate from production uploads. The default
file is `data/evaluation/dataset.jsonl`; it is intentionally empty until
questions and labels are manually verified against source documents.

Each non-comment line must contain:

```json
{
  "question": "Question grounded in an indexed document",
  "reference_answer": "Optional manually written reference answer",
  "relevant_document_ids": ["document UUID"],
  "relevant_chunk_ids": ["chunk UUID"],
  "expected_pages": [1],
  "category": "direct_factual",
  "sufficient_evidence": true
}
```

Sufficient samples must have relevance labels. Insufficient-context samples
may omit them. LLM-generated answers are not accepted as ground truth.

## Retrieval metrics

The runner evaluates dense, BM25, hybrid RRF, and hybrid-plus-reranking under
the same queries and K values.

- Precision@K = relevant results in the first K / K.
- Recall@K = relevant results in the first K / all relevant labels.
- MRR = reciprocal rank of the first relevant result.
- nDCG@K = discounted gain of the returned ranking divided by ideal gain.

Queries without relevant labels have undefined recall and are excluded from
the aggregate recall denominator. Precision, MRR, and nDCG are reported as
zero when no relevance labels exist. This behavior is explicit in the report.

## Latency

High-resolution monotonic timers measure retrieval stages and total query
latency. Reports include average, median, p95, and sample count. Measurements
are local warm/cold mixed observations and must not be interpreted as
production-scale capacity tests.

## Commands

Start the backend, then run:

```powershell
Invoke-WebRequest -Method POST http://127.0.0.1:8000/evaluation/run `
  -ContentType 'application/json' `
  -Body '{"k_values":[5,10]}'
Invoke-WebRequest http://127.0.0.1:8000/evaluation/runs
Invoke-WebRequest http://127.0.0.1:8000/evaluation/metrics
```

The Streamlit Evaluation page provides the same run and historical-report
workflow.
