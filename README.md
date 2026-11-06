# Enterprise Knowledge Intelligence System

This project provides document ingestion, persistent semantic and keyword
retrieval, grounded Gemini answers, structured citations, conversation memory,
and reproducible retrieval evaluation.

## Stack

FastAPI, SQLite/SQLAlchemy, PyMuPDF, python-docx, Sentence Transformers,
FAISS, BM25, Gemini, and Streamlit.

## Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```powershell
streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Set `GEMINI_API_KEY` in the environment for grounded answer generation. Model
and retrieval settings are configured through the existing `.env` settings.
Secrets are never displayed by the frontend.

## API

- `GET /health`
- `POST /documents/upload`
- `GET /documents`
- `GET /documents/{document_id}`
- `GET /documents/{document_id}/status`
- `DELETE /documents/{document_id}`
- `POST /chat`
- `POST /evaluation/run`
- `GET /evaluation/runs`
- `GET /evaluation/runs/{run_id}`
- `GET /evaluation/metrics`

FastAPI OpenAPI documentation is available at `/docs`.

## Evaluation

Add manually verified JSONL records to `data/evaluation/dataset.jsonl`, then
run the evaluation endpoints or use the Streamlit Evaluation page. See
[EVALUATION.md](./EVALUATION.md) for the schema, metric definitions, and
limitations. See [ARCHITECTURE.md](./ARCHITECTURE.md) for the data flow and
[LIMITATIONS.md](./LIMITATIONS.md) for unresolved technical debt.
