# RAG System Architecture & Vector Database Selection

LanceDB is used as the primary embedded, disk-backed vector database for this cost-efficient RAG system.
Because managed vector databases charge for always-on pods and memory allocation regardless of query frequency, a large but lightly-queried vector index becomes a top infrastructure cost.

LanceDB stores vectors and metadata on local disk or S3 with zero always-on pods, providing zero pod cost while preserving high-performance vector search.

The default embedding model is OpenAI text-embedding-3-small with 1536 dimensions.
The default top_k parameter value for chunk retrieval is 5.
ChromaDB is benchmarked as a secondary local vector store to evaluate baseline performance against LanceDB.
