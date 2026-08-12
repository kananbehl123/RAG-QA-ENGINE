from src.eval.retrieval_eval import RetrievalEvaluator
from src.eval.answer_eval import AnswerEvaluator


def test_retrieval_metrics():
    retrieved = [
        {"chunk_id": "chunk_1"},
        {"chunk_id": "chunk_2"},
        {"chunk_id": "chunk_3"}
    ]
    gold = ["chunk_2"]

    hit_rate = RetrievalEvaluator.calculate_hit_rate([c["chunk_id"] for c in retrieved], set(gold), k=3)
    assert hit_rate == 1.0

    mrr = RetrievalEvaluator.calculate_mrr([c["chunk_id"] for c in retrieved], set(gold))
    assert mrr == 0.5  # 1/2 because chunk_2 is at rank 2


def test_answer_metrics():
    pred = "LanceDB is a disk-backed zero-pod vector database [Source 1]."
    gold = "LanceDB is an embedded disk-backed vector store."
    chunks = [{"text": "LanceDB is a disk-backed zero-pod vector database."}]

    metrics = AnswerEvaluator.evaluate_answer(pred, gold, "What is LanceDB?", chunks)
    assert metrics["exact_match"] == 0.0 or metrics["exact_match"] == 1.0
    assert metrics["f1_score"] >= 0.5
    assert metrics["faithfulness"] > 0.5
    assert metrics["answer_relevance"] > 0.5
