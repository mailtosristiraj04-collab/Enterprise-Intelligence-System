# Ground-truth dataset preparation

The evaluation dataset must contain human-verified questions about documents
that are actually indexed in this installation. Do not copy IDs from examples,
invent records, or use an LLM to create labels. The repository keeps
`data/evaluation/dataset.jsonl` empty until this process has been completed.

## Target set

Build 30–50 questions across the following categories:

| Category | Suggested count | What to verify |
|---|---:|---|
| `direct_factual` | 8–12 | One chunk contains the answer |
| `multi_chunk` | 6–10 | The answer requires two or more chunks |
| `multi_document` | 4–8 | The answer requires chunks from two or more documents |
| `semantic_paraphrase` | 4–6 | The question uses different wording from the source |
| `technical_terminology` | 3–5 | The question tests domain-specific terms |
| `conflicting` | 2–4 | Sources contain genuinely conflicting statements |
| `insufficient` | 5–8 | The indexed documents do not support an answer |

These are planning ranges, not records to insert automatically. Every
question, answer, and label must be reviewed by a person against the current
document text.

## JSONL contract

Every answerable record must contain all of these fields:

```json
{
  "question": "MANUALLY WRITTEN QUESTION",
  "reference_answer": "MANUALLY VERIFIED ANSWER",
  "relevant_document_ids": ["REAL DOCUMENT UUID"],
  "relevant_chunk_ids": ["REAL CHUNK UUID", "ANOTHER REAL CHUNK UUID"],
  "expected_pages": [1],
  "category": "direct_factual",
  "sufficient_evidence": true
}
```

For an intentionally unanswerable record, use no answer or relevance labels:

```json
{
  "question": "MANUALLY WRITTEN QUESTION NOT ANSWERED BY THE INDEX",
  "category": "insufficient",
  "sufficient_evidence": false
}
```

The examples above contain placeholders and are templates only. They must not
be copied into the production dataset. In particular, IDs must come from the
database, and an insufficient sample must not contain a reference answer,
document IDs, chunk IDs, or pages.

## Finding real IDs

First index the source documents through the normal API or Knowledge Base UI.
Then inspect the same SQLite database configured for the backend. Run this
read-only query from the project root:

```powershell
@'
from backend.database.database import get_database
from backend.database.models import Chunk, Document

database = get_database()
database.create_tables()
session = database.get_session()
try:
    for document in session.query(Document).order_by(Document.uploaded_at):
        print(f"DOCUMENT {document.id} | {document.original_filename}")
        for chunk in session.query(Chunk).filter_by(document_id=document.id).order_by(Chunk.chunk_index):
            print(f"  CHUNK {chunk.id} | page={chunk.page_number} | {chunk.text}")
finally:
    session.close()
'@ | python -
```

Copy an ID only after reading the corresponding chunk text and confirming that
it supports the manually written answer. If the answer uses two chunks, list
both chunk IDs. If it uses multiple documents, list every supporting document
and chunk ID.

## Review procedure by question type

### Direct factual

Read one chunk, write a question whose answer is explicitly present, and
record that chunk's real ID and page:

```json
{
  "question": "QUESTION ANSWERED BY ONE VERIFIED CHUNK",
  "reference_answer": "ANSWER QUOTED OR FAITHFULLY PARAPHRASED FROM THAT CHUNK",
  "relevant_document_ids": ["REAL DOCUMENT UUID"],
  "relevant_chunk_ids": ["REAL CHUNK UUID"],
  "expected_pages": [REAL_PAGE_NUMBER],
  "category": "direct_factual",
  "sufficient_evidence": true
}
```

### Multi-chunk

Confirm that no single labeled chunk is sufficient and that the answer
requires combining the labeled chunks. Include every required chunk ID and
all pages that are known:

```json
{
  "question": "QUESTION REQUIRING MULTIPLE VERIFIED CHUNKS",
  "reference_answer": "MANUALLY VERIFIED COMBINED ANSWER",
  "relevant_document_ids": ["REAL DOCUMENT UUID"],
  "relevant_chunk_ids": ["REAL CHUNK UUID", "REAL CHUNK UUID"],
  "expected_pages": [REAL_PAGE_NUMBER, REAL_PAGE_NUMBER],
  "category": "multi_chunk",
  "sufficient_evidence": true
}
```

### Multi-document

Verify each document independently and record the chunks that must be
combined. Every labeled chunk's parent document must appear in
`relevant_document_ids`:

```json
{
  "question": "QUESTION REQUIRING VERIFIED FACTS FROM TWO DOCUMENTS",
  "reference_answer": "MANUALLY VERIFIED SYNTHESIS",
  "relevant_document_ids": ["REAL DOCUMENT UUID", "REAL DOCUMENT UUID"],
  "relevant_chunk_ids": ["REAL CHUNK UUID", "REAL CHUNK UUID"],
  "expected_pages": [REAL_PAGE_NUMBER, REAL_PAGE_NUMBER],
  "category": "multi_document",
  "sufficient_evidence": true
}
```

### Unanswerable

Search all indexed documents for the subject. Only mark the question
insufficient after a human review confirms that the answer is not supported.
Do not provide a guessed reference answer or “relevant” IDs:

```json
{
  "question": "QUESTION WHOSE ANSWER IS ABSENT FROM THE INDEX",
  "category": "insufficient",
  "sufficient_evidence": false
}
```

## Validation workflow

1. Save manually reviewed records as JSONL in the evaluation directory.
2. Run the dataset loader against the configured database before benchmarking.
3. Fix every validation error; do not bypass it by removing labels.
4. Review the source text again after any document re-ingestion because chunk
   IDs can change.
5. Run the evaluation API only after all records pass validation.

The loader rejects missing document IDs, missing chunk IDs, chunks whose
parent document is not labeled, invalid pages, duplicate labels, answerable
records without manual reference answers, and insufficient records that carry
answer labels.
