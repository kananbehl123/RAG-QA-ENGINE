# Vector Database Cost Comparison Analysis

## Infrastructure Cost at Scale (100K, 1M, 10M Vectors)

| Scale   | Disk Storage   | LanceDB (Embedded)   | Pinecone (Managed)   | Qdrant Cloud       | LanceDB Savings   |
|---------|----------------|----------------------|----------------------|--------------------|-------------------|
| 100K    | 0.619 GB       | $15.05               | $70.00 (1 pods)      | $45.00 (1 nodes)   | $54.95 (78.5%)    |
| 1M      | 6.188 GB       | $15.50               | $140.00 (2 pods)     | $90.00 (2 nodes)   | $124.50 (88.9%)   |
| 10M     | 61.877 GB      | $19.95               | $1400.00 (20 pods)   | $765.00 (17 nodes) | $1380.05 (98.6%)  |

## Key Cost Assumptions & Trade-offs

1. **Dimensionality & Chunk Size**:
   - Embedding Model: `text-embedding-3-small` (1536 dimensions, 6,144 bytes raw vector).
   - Text & Metadata payload: ~500 bytes per chunk (~6.64 KB total stored footprint per vector chunk).

2. **Why LanceDB achieves 90%+ Cost Reduction**:
   - **Zero Always-On Pods**: Managed vector databases (Pinecone, Qdrant Cloud, Weaviate) keep HNSW index nodes resident in RAM 24/7, costing $45 - $70/month per node regardless of query volume.
   - **Disk-Backed Columnar Storage**: LanceDB stores vectors and metadata on disk (local EBS / S3) with zero-copy disk scanning, reducing RAM footprint by over 80%.

3. **Trade-offs Accepted**:
   - **Indexing / Query Latency at Ultra-High Scale**: Full RAM-based HNSW achieves sub-10ms latency at 10M+ scale, whereas disk-backed LanceDB trades 15-30ms p95 latency for massive infra cost savings.
   - **Horizontal Multi-Node Clustering**: LanceDB is optimized for single-instance embedded disk/S3 operation. For multi-region serverless scaling, managed services offer built-in cross-region replication.
