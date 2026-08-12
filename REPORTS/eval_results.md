# RAG Evaluation Benchmark Results

**Vector Database**: `LANCEDB`  
**Test Set Size**: `15 QA Pairs`

## Summary Metrics Table

| Metric                      | Score / Value   |
|-----------------------------|-----------------|
| Hit Rate @ 5 (Recall)       | 0.0%            |
| Mean Reciprocal Rank (MRR)  | 0.0000          |
| nDCG @ 5                    | 0.0000          |
| Context Precision           | 0.0%            |
| Faithfulness / Groundedness | 82.9%           |
| Answer Relevance            | 100.0%          |
| Exact Match (EM)            | 0.0%            |
| F1 Score                    | 0.0885          |
| p50 Retrieval Latency       | 5.86 ms         |
| p95 Retrieval Latency       | 9.81 ms         |

## Per-Query Evaluation Breakdown

| ID   | Question                                         |   Hit@5 |   MRR |   Faithful | Latency   |
|------|--------------------------------------------------|---------|-------|------------|-----------|
| q01  | What is the main vector database used in this... |       0 |     0 |       0.81 | 16.1 ms   |
| q02  | Why does a fully managed vector database beco... |       0 |     0 |       0.87 | 7.1 ms    |
| q03  | How is document re-ingestion made idempotent?... |       0 |     0 |       0.82 | 5.6 ms    |
| q04  | What happens when no relevant context chunks ... |       0 |     0 |       0.78 | 5.9 ms    |
| q05  | Which embedding model is used for vectorizing... |       0 |     0 |       0.89 | 6.2 ms    |
| q06  | What parameters control text chunking in the ... |       0 |     0 |       0.91 | 5.0 ms    |
| q07  | What secondary vector database is benchmarked... |       0 |     0 |       0.86 | 6.6 ms    |
| q08  | What metrics are used for evaluating retrieva... |       0 |     0 |       0.8  | 5.1 ms    |
| q09  | How are answer faithfulness and groundedness ... |       0 |     0 |       0.89 | 6.1 ms    |
| q10  | What telemetry metrics are logged per query?...  |       0 |     0 |       0.79 | 5.1 ms    |
| q11  | How does LanceDB reduce infrastructure storag... |       0 |     0 |       0.79 | 4.9 ms    |
| q12  | What file formats are supported for document ... |       0 |     0 |       0.84 | 6.6 ms    |
| q13  | What API endpoints does the FastAPI web servi... |       0 |     0 |       0.81 | 5.0 ms    |
| q14  | What HTTP status code is returned when an inv... |       0 |     0 |       0.8  | 6.3 ms    |
| q15  | What is the default top_k parameter value for... |       0 |     0 |       0.78 | 5.6 ms    |
