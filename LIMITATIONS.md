# Known limitations

- The default evaluation dataset is empty. Results are not reported until
  manually verified labels are added.
- Semantic answer faithfulness, relevance, and completeness require a
  configured Gemini judge and are not silently assigned perfect scores.
- Gemini network integration remains unverified when `GEMINI_API_KEY` is not
  configured.
- SQLite, FAISS, and BM25 do not share an atomic transaction. Ingestion and
  deletion perform ordered cleanup and rely on rebuild/recovery checks.
- Synchronous extraction, embedding, reranking, and evaluation calls are
  suitable for the current single-process deployment but limit concurrent
  throughput for large workloads.
- Latency reports are local measurements, not capacity or production
  benchmarks.
- Scanned PDFs require OCR, which is not currently included.
- Evaluation API runs are synchronous; a long dataset can occupy a request
  worker. A background job system was intentionally not introduced.
