import re
from typing import List, Dict, Any
from src.config import settings


class AnswerEvaluator:
    """
    Evaluates Answer Quality using Groundedness, Relevance, Exact Match (EM), and F1 Score metrics.
    """

    @staticmethod
    def calculate_exact_match(prediction: str, ground_truth: str) -> float:
        norm_pred = AnswerEvaluator._normalize_text(prediction)
        norm_gt = AnswerEvaluator._normalize_text(ground_truth)
        return 1.0 if norm_pred == norm_gt or norm_gt in norm_pred else 0.0

    @staticmethod
    def calculate_f1_score(prediction: str, ground_truth: str) -> float:
        pred_tokens = AnswerEvaluator._normalize_text(prediction).split()
        gt_tokens = AnswerEvaluator._normalize_text(ground_truth).split()

        if not pred_tokens or not gt_tokens:
            return 0.0

        common = set(pred_tokens) & set(gt_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def calculate_faithfulness(answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        Evaluates whether answer statements cite sources and derive strictly from context.
        """
        if not context_chunks:
            return 1.0 if "cannot find relevant information" in answer.lower() else 0.0

        # Check citation presence
        has_citations = bool(re.search(r"\[Source \d+\]", answer)) or "Source" in answer
        score = 0.5 if has_citations else 0.2

        # Check semantic token overlap with retrieved contexts
        context_text = " ".join(c["text"] for c in context_chunks).lower()
        answer_words = set(AnswerEvaluator._normalize_text(answer).split())
        context_words = set(AnswerEvaluator._normalize_text(context_text).split())

        overlap_ratio = len(answer_words & context_words) / len(answer_words) if answer_words else 0.0
        score += 0.5 * overlap_ratio

        return min(1.0, round(score, 2))

    @staticmethod
    def calculate_answer_relevance(answer: str, question: str) -> float:
        """
        Evaluates keyword alignment between generated answer and question intent.
        """
        q_words = set(AnswerEvaluator._normalize_text(question).split()) - {"what", "is", "the", "how", "why", "which", "where"}
        a_words = set(AnswerEvaluator._normalize_text(answer).split())

        if not q_words:
            return 1.0

        overlap = len(q_words & a_words) / len(q_words)
        return round(min(1.0, max(0.4, overlap + 0.3)), 2)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    @classmethod
    def evaluate_answer(
        cls,
        prediction: str,
        gold_answer: str,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        return {
            "exact_match": cls.calculate_exact_match(prediction, gold_answer),
            "f1_score": round(cls.calculate_f1_score(prediction, gold_answer), 4),
            "faithfulness": cls.calculate_faithfulness(prediction, retrieved_chunks),
            "answer_relevance": cls.calculate_answer_relevance(prediction, question)
        }
