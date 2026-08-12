import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from src.config import settings
from src.api.schemas import (
    IngestPathRequest, IngestResponse,
    QueryRequest, QueryResponse,
    HealthResponse, MetricsResponse
)
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.search import Retriever
from src.llm.generator import QAEngine
from src.telemetry.logger import telemetry

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Check API health and vector DB status")
def get_health():
    retriever = Retriever()
    count = retriever.store.count()
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        vector_db_type=settings.VECTOR_DB_TYPE,
        stored_chunks_count=count,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Retrieve aggregate performance metrics (p50/p95 latency, tokens)")
def get_metrics():
    stats = telemetry.get_performance_stats()
    return MetricsResponse(**stats)


@router.post("/query", response_model=QueryResponse, summary="Execute RAG QA query over vector store")
def query_rag(req: QueryRequest):
    try:
        engine = QAEngine()
        result = engine.answer_question(
            question=req.question,
            top_k=req.top_k,
            metadata_filter=req.metadata_filter,
            min_score_threshold=req.min_score_threshold or 0.0
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution error: {str(e)}")


@router.post("/ingest/path", response_model=IngestResponse, summary="Ingest document or folder path into vector DB")
def ingest_path(req: IngestPathRequest):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    pipeline = IngestionPipeline()
    if p.is_file():
        res = pipeline.ingest_file(str(p))
        return IngestResponse(
            status="success",
            total_files_processed=1,
            total_chunks=res["total_chunks"],
            inserted_chunks=res["inserted_chunks"],
            skipped_chunks=res["skipped_chunks"],
            details=[res]
        )
    elif p.is_dir():
        results = pipeline.ingest_directory(str(p))
        total_chunks = sum(r.get("total_chunks", 0) for r in results)
        inserted = sum(r.get("inserted_chunks", 0) for r in results)
        skipped = sum(r.get("skipped_chunks", 0) for r in results)
        return IngestResponse(
            status="success",
            total_files_processed=len(results),
            total_chunks=total_chunks,
            inserted_chunks=inserted,
            skipped_chunks=skipped,
            details=results
        )


@router.post("/ingest/file", response_model=IngestResponse, summary="Upload document file (PDF/HTML/MD) directly")
async def ingest_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile("wb", suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = IngestionPipeline()
        res = pipeline.ingest_file(tmp_path)
        # Fix source file name in response details
        res["file"] = file.filename
        return IngestResponse(
            status="success",
            total_files_processed=1,
            total_chunks=res["total_chunks"],
            inserted_chunks=res["inserted_chunks"],
            skipped_chunks=res["skipped_chunks"],
            details=[res]
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
