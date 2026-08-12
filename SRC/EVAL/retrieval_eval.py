import math
from typing import List, Dict, Any, Set


class RetrievalEvaluator:
    """
    Computes standard Information Retrieval (IR) metrics over retrieved context chunks.
    Metrics:
    - Hit Rate @ k (Recall @ k)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain @ k (nDCG @ k)
    - Context Precision @ k
    """

    @staticmethod
    def calculate_hit_rate(retrieved_chunk_ids: List[str], gold_chunk_ids: Set[str], k: int) -> float:
        top_k_ids = retrieved_chunk_ids[:k]
        for c_id in top_k_ids:
            if c_id in gold_chunk_ids or any(g in c_id for g in gold_chunk_ids):
                return 1.0
        return 0.0

    @staticmethod
    def calculate_mrr(retrieved_chunk_ids: List[str], gold_chunk_ids: Set[str]) -> float:
        for rank, c_id in enumerate(retrieved_chunk_ids, start=1):
            if c_id in gold_chunk_ids or any(g in c_id for g in gold_chunk_ids):
                return 1.0 / rank
        return 0.0

    @staticmethod
    def calculate_ndcg(retrieved_chunk_ids: List[str], gold_chunk_ids: Set[str], k: int) -> float:
        top_k_ids = retrieved_chunk_ids[:k]
        dcg = 0.0
        for i, c_id in enumerate(top_k_ids, start=1):
            rel = 1.0 if (c_id in gold_chunk_ids or any(g in c_id for g in gold_chunk_ids)) else 0.0
            dcg += rel / math.log2(i + 1)

        # Ideal DCG
        ideal_rels = [1.0] * min(len(gold_chunk_ids), k)
        idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_rels, start=1))

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def calculate_context_precision(retrieved_chunk_ids: List[str], gold_chunk_ids: Set[str], k: int) -> float:
        top_k_ids = retrieved_chunk_ids[:k]
        if not top_k_ids:
            return 0.0

        hits = 0
        precision_at_i_sum = 0.0
        for i, c_id in enumerate(top_k_ids, start=1):
            is_rel = c_id in gold_chunk_ids or any(g in c_id for g in gold_chunk_ids)
            if is_rel:
                hits += 1
                precision_at_i_sum += hits / i

        return precision_at_i_sum / hits if hits > 0 else 0.0

    @classmethod
    def evaluate_query(
        cls,
        retrieved_chunks: List[Dict[str, Any]],
        gold_chunk_ids: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
        gold_set = set(gold_chunk_ids)

        return {
            "hit_rate_at_k": cls.calculate_hit_rate(retrieved_ids, gold_set, k),
            "mrr": cls.calculate_mrr(retrieved_ids, gold_set),
            "ndcg_at_k": cls.calculate_ndcg(retrieved_ids, gold_set, k),
            "context_precision": cls.calculate_context_precision(retrieved_ids, gold_set, k)
        }
