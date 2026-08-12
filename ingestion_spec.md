# Document Ingestion & Chunking Specification

Document re-ingestion is made idempotent by computing a SHA-256 hash derived from the source file path and chunk content to detect duplicate vectors. If a chunk hash already exists in the vector store, re-indexing is skipped.

Parameters control text chunking in the pipeline:
- Chunk size (default 500 characters)
- Chunk overlap (default 50 characters)

The ingestion engine supports PDF, HTML, and Markdown (.md) document file formats.
