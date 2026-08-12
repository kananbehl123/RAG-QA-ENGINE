import os
import tempfile
import pytest
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunker import TextChunker
from src.embeddings.embedder import EmbeddingManager
from src.ingestion.pipeline import IngestionPipeline
from src.db.lancedb_store import LanceDBStore


def test_markdown_loader_and_chunker():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("# Architecture\n\nThis is a sample document for testing RAG ingestion.\n\n" * 10)
        temp_path = f.name

    try:
        doc = DocumentLoader.load_file(temp_path)
        assert doc["file_type"] == "md"
        assert len(doc["text"]) > 0

        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert "chunk_hash" in chunks[0]
        assert "source_file" in chunks[0]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_idempotent_ingestion(tmp_path):
    # Create temp LanceDB directory
    db_dir = str(tmp_path / "lancedb_test")
    store = LanceDBStore(db_dir)
    embedder = EmbeddingManager(provider="mock", dimension=64)

    pipeline = IngestionPipeline(
        vector_store=store,
        embedder=embedder,
        chunk_size=150,
        chunk_overlap=20
    )

    # Create sample doc
    sample_file = tmp_path / "sample.md"
    sample_file.write_text("# RAG Benchmark\n\nLanceDB provides zero-pod cost-efficient vector storage.")

    # Ingest file first time
    res1 = pipeline.ingest_file(str(sample_file))
    assert res1["status"] == "success"
    assert res1["inserted_chunks"] > 0
    assert res1["skipped_chunks"] == 0

    first_count = store.count()
    assert first_count == res1["inserted_chunks"]

    # Re-ingest exact same file (idempotency check)
    res2 = pipeline.ingest_file(str(sample_file))
    assert res2["status"] == "idempotent_skip"
    assert res2["inserted_chunks"] == 0
    assert res2["skipped_chunks"] == res1["total_chunks"]

    # Verify database vector count remained identical (no duplicates!)
    second_count = store.count()
    assert second_count == first_count
