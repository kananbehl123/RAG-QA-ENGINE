# API Specification & Endpoints

The API endpoints that the FastAPI web service exposes include:
- GET /health (Health check and vector count)
- GET /metrics (Telemetry p50/p95 latency and token metrics)
- POST /query (Execute grounded QA search)
- POST /ingest (Ingest file or directory path)

Passing an invalid path to ingestion returns a 404 Not Found error.
