# Architecture

```text
Streamlit frontend --HTTP--> FastAPI
                              |
                 SQLite <--- ingestion ---> uploads
                   |          |
            chunks/messages   embeddings
                   |          v
             evaluation <--- FAISS + BM25
                              |
                  dense + sparse -> RRF -> reranker
                              |
                    bounded context -> Gemini
                              |
                         answer + citations
```

Documents are extracted, cleaned, chunked, embedded, and persisted to SQLite
and FAISS. BM25 is persisted beside the vector index and rebuilt when its
chunk signature is stale. Retrieval combines dense and sparse rankings using
reciprocal rank fusion, then optionally reranks candidates with a cross
encoder. The RAG service bounds context, treats retrieved text as untrusted
data, and emits structured citations for supplied chunks only.

Conversation messages are stored in SQLite and are scoped by conversation ID.
The Streamlit application communicates only with the public FastAPI contract.
Evaluation reads the same retrieval services and stores immutable JSON reports
in the `evaluation_runs` table; evaluation records are not production
conversation messages.
