from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class IngestPathRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest")
    db_type: Optional[str] = Field(default=None, description="Vector DB backend override (lancedb or chromadb)")


class IngestResponse(BaseModel):
    status: str
    total_files_processed: int
    total_chunks: int
    inserted_chunks: int
    skipped_chunks: int
    details: List[Dict[str, Any]]


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question string")
    top_k: Optional[int] = Field(default=5, description="Number of context chunks to retrieve")
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter dictionary")
    min_score_threshold: Optional[float] = Field(default=0.0, description="Minimum similarity score threshold")


class ChunkCitation(BaseModel):
    citation_id: str
    source_file: str
    chunk_id: str


class RetrievedChunk(BaseModel):
    citation_id: str
    chunk_id: str
    doc_id: str
    source_file: str
    file_type: str
    chunk_index: int
    text: str
    score: float
    distance: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    citations: List[ChunkCitation]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    app_name: str
    vector_db_type: str
    stored_chunks_count: int
    embedding_model: str
    llm_model: str


class MetricsResponse(BaseModel):
    total_queries: int
    p50_retrieval_latency_ms: float
    p95_retrieval_latency_ms: float
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    avg_tokens_per_query: float
