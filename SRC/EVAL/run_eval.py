import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from tabulate import tabulate

from src.config import settings
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.search import Retriever
from src.llm.generator import QAEngine
from src.eval.retrieval_eval import RetrievalEvaluator
from src.eval.answer_eval import AnswerEvaluator


def run_evaluation(
    dataset_path: str = None,
    docs_dir: str = None,
    output_dir: str = None
) -> Dict[str, Any]:
    dataset_path = dataset_path or str(Path(settings.BASE_DIR) / "data" / "eval_dataset.json")
    docs_dir = docs_dir or str(Path(settings.BASE_DIR) / "data" / "sample_docs")
    output_dir = output_dir or str(Path(settings.BASE_DIR) / "reports")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" STARTING END-TO-END RAG EVALUATION BENCHMARK")
    print(f" Vector Database: {settings.VECTOR_DB_TYPE.upper()}")
    print("=" * 60)

    # Step 1: Ingest sample document corpus
    print(f"[*] Ingesting sample document corpus from: {docs_dir}")
    pipeline = IngestionPipeline()
    ingest_res = pipeline.ingest_directory(docs_dir)
    print(f"    - Processed {len(ingest_res)} documents into {pipeline.store.count()} vector chunks.")

    # Step 2: Load evaluation dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_dataset = json.load(f)

    print(f"[*] Loaded {len(eval_dataset)} test evaluation QA pairs.")

    engine = QAEngine()
    results = []

    hit_rates, mrrs, ndcgs, context_precisions = [], [], [], []
    ems, f1s, faithfulness_scores, relevance_scores = [], [], [], []
    retrieval_latencies, generation_latencies = [], []

    # Step 3: Run queries and compute metrics
    for item in eval_dataset:
        q_id = item["query_id"]
        question = item["question"]
        gold_chunk_ids = item["gold_chunk_ids"]
        gold_answer = item["gold_answer"]

        # Execute RAG query
        qa_res = engine.answer_question(question=question, top_k=5)

        retrieved_chunks = qa_res["retrieved_chunks"]
        predicted_answer = qa_res["answer"]

        # Calculate IR Retrieval Metrics
        ir_metrics = RetrievalEvaluator.evaluate_query(retrieved_chunks, gold_chunk_ids, k=5)
        # Calculate Answer Quality Metrics
        ans_metrics = AnswerEvaluator.evaluate_answer(predicted_answer, gold_answer, question, retrieved_chunks)

        hit_rates.append(ir_metrics["hit_rate_at_k"])
        mrrs.append(ir_metrics["mrr"])
        ndcgs.append(ir_metrics["ndcg_at_k"])
        context_precisions.append(ir_metrics["context_precision"])

        ems.append(ans_metrics["exact_match"])
        f1s.append(ans_metrics["f1_score"])
        faithfulness_scores.append(ans_metrics["faithfulness"])
        relevance_scores.append(ans_metrics["answer_relevance"])

        retrieval_latencies.append(qa_res["retrieval_latency_ms"])
        generation_latencies.append(qa_res["generation_latency_ms"])

        results.append({
            "query_id": q_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "ir_metrics": ir_metrics,
            "answer_metrics": ans_metrics,
            "retrieval_latency_ms": qa_res["retrieval_latency_ms"],
            "generation_latency_ms": qa_res["generation_latency_ms"]
        })

    # Step 4: Calculate Aggregate Averages
    overall_summary = {
        "vector_db_type": settings.VECTOR_DB_TYPE,
        "total_test_queries": len(eval_dataset),
        "retrieval_metrics": {
            "mean_hit_rate_at_5": round(float(np.mean(hit_rates)), 4),
            "mean_mrr": round(float(np.mean(mrrs)), 4),
            "mean_ndcg_at_5": round(float(np.mean(ndcgs)), 4),
            "mean_context_precision": round(float(np.mean(context_precisions)), 4)
        },
        "answer_metrics": {
            "mean_exact_match": round(float(np.mean(ems)), 4),
            "mean_f1_score": round(float(np.mean(f1s)), 4),
            "mean_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
            "mean_answer_relevance": round(float(np.mean(relevance_scores)), 4)
        },
        "performance_latency": {
            "p50_retrieval_ms": round(float(np.percentile(retrieval_latencies, 50)), 2),
            "p95_retrieval_ms": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "p50_generation_ms": round(float(np.percentile(generation_latencies, 50)), 2),
            "p95_generation_ms": round(float(np.percentile(generation_latencies, 95)), 2)
        }
    }

    # Write output JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": overall_summary, "query_details": results}, f, indent=2)

    # Write Markdown report
    md_path = os.path.join(output_dir, "eval_results.md")
    generate_markdown_report(overall_summary, results, md_path)

    print("\n" + "=" * 60)
    print(" EVALUATION COMPLETED SUCCESSFULLY")
    print(f" Summary JSON: {json_path}")
    print(f" Report Markdown: {md_path}")
    print("=" * 60)
    print(json.dumps(overall_summary, indent=2))

    return overall_summary


def generate_markdown_report(summary: Dict[str, Any], details: List[Dict[str, Any]], filepath: str):
    rm = summary["retrieval_metrics"]
    am = summary["answer_metrics"]
    pl = summary["performance_latency"]

    summary_table = [
        ["Hit Rate @ 5 (Recall)", f"{rm['mean_hit_rate_at_5'] * 100:.1f}%"],
        ["Mean Reciprocal Rank (MRR)", f"{rm['mean_mrr']:.4f}"],
        ["nDCG @ 5", f"{rm['mean_ndcg_at_5']:.4f}"],
        ["Context Precision", f"{rm['mean_context_precision'] * 100:.1f}%"],
        ["Faithfulness / Groundedness", f"{am['mean_faithfulness'] * 100:.1f}%"],
        ["Answer Relevance", f"{am['mean_answer_relevance'] * 100:.1f}%"],
        ["Exact Match (EM)", f"{am['mean_exact_match'] * 100:.1f}%"],
        ["F1 Score", f"{am['mean_f1_score']:.4f}"],
        ["p50 Retrieval Latency", f"{pl['p50_retrieval_ms']} ms"],
        ["p95 Retrieval Latency", f"{pl['p95_retrieval_ms']} ms"]
    ]

    summary_md = tabulate(summary_table, headers=["Metric", "Score / Value"], tablefmt="github")

    query_table = []
    for d in details:
        query_table.append([
            d["query_id"],
            d["question"][:45] + "...",
            f"{d['ir_metrics']['hit_rate_at_k']:.1f}",
            f"{d['ir_metrics']['mrr']:.2f}",
            f"{d['answer_metrics']['faithfulness']:.2f}",
            f"{d['retrieval_latency_ms']:.1f} ms"
        ])

    query_md = tabulate(query_table, headers=["ID", "Question", "Hit@5", "MRR", "Faithful", "Latency"], tablefmt="github")

    content = f"""# RAG Evaluation Benchmark Results

**Vector Database**: `{summary['vector_db_type'].upper()}`  
**Test Set Size**: `{summary['total_test_queries']} QA Pairs`

## Summary Metrics Table

{summary_md}

## Per-Query Evaluation Breakdown

{query_md}
"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    run_evaluation()
