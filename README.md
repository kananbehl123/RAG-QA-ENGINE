# Cost-Optimized RAG Application & Benchmarking Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![LanceDB](https://img.shields.io/badge/LanceDB-0.6%2B-orange.svg)](https://lancedb.github.io/lancedb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready Retrieval-Augmented Generation (RAG) system built around the lightweight and cost-efficient **LanceDB** vector database. The project combines grounded question answering with retrieval evaluation, answer-quality assessment, latency monitoring, and infrastructure cost analysis across different vector-storage scales.

---

## 📌 Problem Statement & Motivation

Managed vector databases such as Pinecone Pods, Managed Qdrant, and Weaviate Cloud typically incur infrastructure costs that increase with the number of stored vectors. This is largely due to continuously running RAM-based infrastructure, which can become expensive when the vector index is large but query traffic is relatively low.

This project investigates **LanceDB as a cost-conscious alternative** by combining empirical benchmarking with infrastructure cost modeling. The results indicate that disk-backed vector storage can significantly reduce infrastructure expenses while maintaining strong top-k retrieval performance.

---

## 🏗️ System Architecture

The application follows a complete RAG pipeline:

```text
                    +-----------------------------------+
                    |      Document Collection          |
                    |          PDF / HTML / MD          |
                    +-----------------------------------+
                                    |
                                    v
                    +-----------------------------------+
                    |         Document Ingestion        |
                    |-----------------------------------|
                    | • Configurable Chunking            |
                    | • Chunk Overlap                    |
                    | • SHA-256 Content Hashing          |
                    | • Duplicate-Safe Re-ingestion      |
                    +-----------------------------------+
                                    |
                                    v
                    +-----------------------------------+
                    | Embedding & Metadata Generation   |
                    |-----------------------------------|
                    | • text-embedding-3-small          |
                    | • 1536-dimensional embeddings     |
                    | • Source & file metadata           |
                    | • Chunk hash & timestamp           |
                    +-----------------------------------+
                                    |
                                    v
                  +-----------------+------------------+
                  |                                    |
                  v                                    v
        +---------------------+             +----------------------+
        |   Primary Storage   |             | Benchmark Storage    |
        |      LanceDB       |             |      ChromaDB        |
        |     Embedded DB    |             |      Local DB         |
        +---------------------+             +----------------------+
                  |
                  v
        +------------------------------------------------+
        |          Top-k Similarity Retrieval             |
        |        + Metadata-Based Filtering               |
        +------------------------------------------------+
                              |
                              v
        +------------------------------------------------+
        |          Grounded LLM Response Generation       |
        |------------------------------------------------|
        | • Source Citations                              |
        | • Context-Based Answering                       |
        | • Out-of-Scope Query Handling                   |
        | • Latency & Token Monitoring                    |
        +------------------------------------------------+
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
        +-------------------+     +-------------------------+
        | FastAPI / CLI     |     | Evaluation & Cost       |
        | Interfaces        |     | Benchmarking Framework   |
        +-------------------+     +-------------------------+
```

---

# ⚡ Getting Started

## 1. Installation & Configuration

Clone the repository:

```bash
git clone https://github.com/your-username/cost-efficient-rag.git
cd cost-efficient-rag
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Create your environment configuration:

```bash
cp .env.example .env
```

Configure `.env` as follows:

```env
OPENAI_API_KEY=your_openai_api_key_here
VECTOR_DB_TYPE=lancedb
EMBEDDING_MODEL=text-embedding-3-small
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

## 2. Command-Line Interface

### A. Add Documents to the Knowledge Base

```bash
python -m src.cli ingest --path ./data/sample_docs
```

The ingestion pipeline processes the supplied documents and stores their vector representations.

### B. Ask a Question

```bash
python -m src.cli query --question "Why does a managed vector database become expensive?" --top-k 5
```

The system retrieves the most relevant chunks and generates a grounded response.

### C. View System Statistics

```bash
python -m src.cli stats
```

This displays system-level telemetry and vector-store statistics.

---

# 🌐 FastAPI REST API

Launch the API server using:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Once running, the interactive API documentation can be accessed through the local `/docs` endpoint.

### Available Endpoints

| Method | Endpoint              | Purpose                                    |
| ------ | --------------------- | ------------------------------------------ |
| GET    | `/api/v1/health`      | Check service health and vector counts     |
| GET    | `/api/v1/metrics`     | View latency and token statistics          |
| POST   | `/api/v1/query`       | Submit questions and receive cited answers |
| POST   | `/api/v1/ingest/path` | Ingest a file or directory                 |

---

# 🧪 Testing & Evaluation

Run the automated test suite:

```bash
pytest -v
```

Run the complete RAG evaluation:

```bash
python -m src.eval.run_eval
```

The evaluation framework measures:

* Hit Rate
* Mean Reciprocal Rank (MRR)
* nDCG
* Context Precision
* Faithfulness
* Answer Relevance
* F1 Score
* Retrieval latency

The infrastructure cost calculator can be executed with:

```bash
python -m src.eval.cost_model
```

---

# 📊 Benchmark Results

## 1. Retrieval & Answer Quality

The system was evaluated using a fixed benchmark containing **15 question-answer pairs**.

| Evaluation Category | Metric                |    LanceDB |   ChromaDB |
| ------------------- | --------------------- | ---------: | ---------: |
| Retrieval           | Recall@5 / Hit Rate   | **100.0%** | **100.0%** |
| Retrieval           | MRR                   | **0.9333** |     0.9167 |
| Retrieval           | nDCG@5                | **0.9524** |     0.9410 |
| Retrieval           | Context Precision     |  **91.2%** |      89.5% |
| Answer Quality      | Faithfulness          |  **94.8%** |      93.2% |
| Answer Quality      | Answer Relevance      |  **96.5%** |      95.0% |
| Answer Quality      | F1 Score              | **0.7840** |     0.7650 |
| Performance         | p50 Retrieval Latency | **4.2 ms** |    12.1 ms |
| Performance         | p95 Retrieval Latency | **8.5 ms** |    22.4 ms |

The results show that both databases achieved perfect Recall@5 on the benchmark, while LanceDB demonstrated stronger ranking metrics, answer-quality scores, and retrieval latency.

---

# 💰 Infrastructure Cost Analysis

The cost model evaluates three vector-storage scales:

* 100K vectors
* 1M vectors
* 10M vectors

The embedding dimension is **1536**, based on `text-embedding-3-small`, with an estimated total footprint of approximately **6.64 KB per chunk**.

| Vector Count |  Storage |       LanceDB |  Pinecone | Qdrant Cloud | LanceDB Savings |
| -----------: | -------: | ------------: | --------: | -----------: | --------------: |
|         100K |  0.62 GB | **$15.05/mo** |    $70/mo |       $45/mo |       **78.5%** |
|           1M |  6.18 GB | **$15.50/mo** |   $140/mo |       $90/mo |       **88.9%** |
|          10M | 61.84 GB | **$19.95/mo** | $1,400/mo |      $900/mo |       **98.6%** |

The modeled savings become increasingly significant as the vector collection grows.

---

# 💬 Engineering Discussion

## When should a managed vector database be preferred?

LanceDB is not intended to replace managed vector infrastructure in every workload. A managed solution may become more appropriate when the application requires:

* Extremely high write throughput with many concurrent readers.
* Large-scale distributed retrieval across multiple regions.
* Very low latency requirements at extremely large vector counts.
* Fully managed infrastructure with minimal operational responsibility.
* Built-in cross-region availability and failover.

---

## Retrieval vs. Generation

The evaluation suggests that **generation was more reliable than retrieval**.

The grounded generation pipeline achieved:

* **94.8% faithfulness**
* **96.5% answer relevance**
* Strong handling of out-of-domain queries

However, retrieval remains an area with room for improvement. Although Recall@5 reached **100%**, Context Precision was **91.2%**, indicating that some retrieved chunks were not perfectly relevant.

Potential improvements include:

* Hybrid Dense + BM25 retrieval
* Cross-Encoder re-ranking
* Improved chunk boundary strategies
* More advanced retrieval filtering

---

## Engineering Trade-offs

The system intentionally makes several engineering trade-offs.

### Disk Storage vs. RAM

Instead of relying entirely on in-memory vector indexes, the architecture uses disk-backed Lance storage. This increases retrieval latency slightly but reduces infrastructure requirements.

### Embedded Database vs. Distributed Infrastructure

LanceDB provides an embedded storage model that avoids the continuous infrastructure cost associated with always-running managed vector database nodes.

The project therefore prioritizes **cost efficiency and practical deployment** over extremely low-latency distributed workloads.

---

# 🛠️ Repository Structure

```text
.
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── src/
│   ├── config.py
│   ├── cli.py
│   │
│   ├── api/
│   │   ├── app.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── db/
│   │   ├── base.py
│   │   ├── lancedb_store.py
│   │   └── chroma_store.py
│   │
│   ├── embeddings/
│   │   └── embedder.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── loaders.py
│   │   └── pipeline.py
│   │
│   ├── llm/
│   │   ├── generator.py
│   │   └── prompts.py
│   │
│   ├── retrieval/
│   │   └── search.py
│   │
│   ├── eval/
│   │   ├── retrieval_eval.py
│   │   ├── answer_eval.py
│   │   ├── cost_model.py
│   │   └── run_eval.py
│   │
│   └── telemetry/
│       └── logger.py
│
├── data/
│   ├── sample_docs/
│   └── eval_dataset.json
│
├── reports/
│   ├── eval_results.json
│   ├── eval_results.md
│   └── cost_analysis.md
│
└── tests/
```

---

# 🎯 Project Summary

This project demonstrates a complete **cost-aware RAG architecture** that combines document ingestion, vector embeddings, semantic retrieval, grounded LLM generation, API deployment, automated evaluation, telemetry, and infrastructure cost modeling.

The benchmark compares **LanceDB and ChromaDB**, while the cost model evaluates how disk-backed vector storage can reduce infrastructure expenditure as vector collections scale.

The overall goal is to provide a practical RAG implementation that balances **retrieval quality, response reliability, latency, scalability, and infrastructure cost**.
