# Multimodal RAG Engine — SPEC

## Goal
Build a Python FastAPI service that ingests documents (PDF, DOCX, PNG, JPG), detects blocks (text/table/image), routes each block:
- TEXT -> verbatim extraction (no summarization at ingest)
- TABLE -> structured JSON extraction (rows/columns) + summary
- DIAGRAM / FLOWCHART / MATRIX (images) -> Vision LLM analysis via OpenRouter, returning strict structured JSON

Produce outputs selectable as `txt` or `md`. Store nodes (blocks) with embeddings into Qdrant. Optionally forward the final composed output (md/txt) to an external RAG endpoint via POST.

## Requirements (high-level)
- Python 3.11+
- FastAPI + Uvicorn
- OpenRouter for LLM/Vision (model selectable via request)
- Local embeddings using sentence-transformers (default `all-mpnet-base-v2`)
- Qdrant as vector DB (default connection via env)
- Docker + docker-compose
- Config via `.env` (OPENROUTER_API_KEY, QDRANT_URL, MODEL_DEFAULT, etc.)
- All LLM responses for vision tasks must be **strict JSON** following provided schema

## API Endpoints
- `POST /ingest` — multipart upload `file`, optional `model_id`, `output_format` (`txt`|`md`), optional `forward_to` JSON (`{url, api_key}`)
  - Response: `202 Accepted` + `{job_id, status_url}`
  - Processing happens in background (FastAPI BackgroundTasks)
- `GET /status/{job_id}` — job status and result metadata
- `GET /result/{job_id}?format=txt|md` — retrieve composed output if ready
- `POST /forward/{job_id}` — manual forward to external RAG (retries supported)
- `POST /reindex/{doc_id}` — reprocess document with new model/policy

## Processing pipeline (per document)
1. Save upload
2. Layout detection (pdfplumber/pdf2image for PDF, python-docx for DOCX, PIL/OpenCV for images)
3. Block classification: text | table | image (diagram/matrix)
4. Processing:
   - Text: verbatim extract -> embed
   - Table: parse rows/cols -> JSON summary -> embed
   - Image (diagram/matrix): send to Vision LLM -> strict JSON -> create human summary -> embed + optional image embedding
5. Store each node (node_id, doc_id, page, block_type, content_text, structured_json, metadata, embedding) into Qdrant
6. Compose final output (txt/md) using page order:
   - Text blocks inserted verbatim
   - Table blocks: table-text + summary
   - Diagram blocks: summary + codeblock of structured JSON
7. Optional forward to external RAG via POST `{job_id, doc_id, format, content_md, metadata}`

## JSON Schemas (vision outputs)
- Flowchart:
```json
{
 "type": "flowchart",
 "confidence": 0.0,
 "nodes": [{"id":"n1","label":"Detect Event","text":"..."}],
 "edges": [{"from":"n1","to":"n2","label":"if high"}],
 "summary": "..."
}
