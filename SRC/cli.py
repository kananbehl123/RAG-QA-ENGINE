import sys
import argparse
import json
from pathlib import Path
from tabulate import tabulate
from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.search import Retriever
from src.llm.generator import QAEngine
from src.telemetry.logger import telemetry


def main():
    parser = argparse.ArgumentParser(
        description="Cost-Efficient RAG CLI Application (LanceDB / ChromaDB)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest document or directory")
    ingest_parser.add_argument(
        "--path", required=True, help="File or directory path to ingest"
    )
    ingest_parser.add_argument(
        "--db", choices=["lancedb", "chromadb"], default=settings.VECTOR_DB_TYPE,
        help="Vector store backend (default: lancedb)"
    )

    # Query Command
    query_parser = subparsers.add_parser("query", help="Query the RAG QA engine")
    query_parser.add_argument(
        "--question", "-q", required=True, help="User question string"
    )
    query_parser.add_argument(
        "--top-k", "-k", type=int, default=settings.TOP_K, help="Number of chunks to retrieve"
    )
    query_parser.add_argument(
        "--filter-type", help="Optional metadata filter by file extension (e.g. pdf, md, html)"
    )

    # Stats Command
    stats_parser = subparsers.add_parser("stats", help="Display vector database & telemetry stats")

    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = IngestionPipeline()
        p = Path(args.path)
        if p.is_file():
            print(f"[*] Ingesting file: {args.path} ...")
            res = pipeline.ingest_file(args.path)
            print(json.dumps(res, indent=2))
        elif p.is_dir():
            print(f"[*] Ingesting directory: {args.path} ...")
            results = pipeline.ingest_directory(args.path)
            table_data = [
                [r["file"], r["status"], r["total_chunks"], r["inserted_chunks"], r["skipped_chunks"]]
                for r in results
            ]
            print("\n" + tabulate(
                table_data,
                headers=["File", "Status", "Total Chunks", "Inserted", "Skipped"],
                tablefmt="grid"
            ))
        else:
            print(f"[!] Path does not exist: {args.path}")
            sys.exit(1)

    elif args.command == "query":
        print(f"[*] Searching & Generating Answer for: '{args.question}'...")
        engine = QAEngine()
        meta_filter = {"file_type": args.filter_type} if args.filter_type else None
        res = engine.answer_question(question=args.question, top_k=args.top_k, metadata_filter=meta_filter)

        print("\n" + "=" * 60)
        print(f"QUESTION: {res['question']}")
        print("=" * 60)
        print(f"ANSWER:\n{res['answer']}\n")
        print("-" * 60)
        print(f"PERFORMANCE TELEMETRY:")
        print(f"  - Total Latency: {res['total_latency_ms']} ms")
        print(f"  - Retrieval Latency: {res['retrieval_latency_ms']} ms")
        print(f"  - Generation Latency: {res['generation_latency_ms']} ms")
        print(f"  - Token Usage: {res['token_usage']}")

        if res["retrieved_chunks"]:
            print("\nRETRIEVED CITATION CHUNKS:")
            for idx, c in enumerate(res["retrieved_chunks"], 1):
                print(f"  [{idx}] {c['citation_id']} -> Score: {c['score']} | {c['source_file']} (Chunk {c['chunk_index']})")
        print("=" * 60)

    elif args.command == "stats":
        retriever = Retriever()
        count = retriever.store.count()
        stats = telemetry.get_performance_stats()
        print("\n" + "=" * 50)
        print(f" RAG SYSTEM STATISTICS ({settings.VECTOR_DB_TYPE.upper()})")
        print("=" * 50)
        print(f" Total Stored Chunks: {count}")
        print(f" Total Logged Queries: {stats['total_queries']}")
        print(f" p50 Retrieval Latency: {stats['p50_retrieval_latency_ms']} ms")
        print(f" p95 Retrieval Latency: {stats['p95_retrieval_latency_ms']} ms")
        print(f" p50 Total Latency:     {stats['p50_total_latency_ms']} ms")
        print(f" p95 Total Latency:     {stats['p95_total_latency_ms']} ms")
        print(f" Avg Tokens / Query:    {stats['avg_tokens_per_query']}")
        print("=" * 50 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
