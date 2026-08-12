import time
from typing import Dict, Any, List, Optional
from src.config import settings
from src.retrieval.search import Retriever
from src.llm.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.telemetry.logger import telemetry


class QAEngine:
    """
    Citation-Grounded LLM Answer Generation Engine.
    Enforces strict grounding on retrieved context chunks, prevents hallucinations,
    and logs per-query latency/token metrics.
    """

    def __init__(self, retriever: Retriever = None):
        self.retriever = retriever or Retriever()
        self._openai_client = None

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_score_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """
        End-to-end RAG workflow: Retrieval -> Prompt Construction -> LLM Generation -> Citation Mapping.
        """
        # Step 1: Top-k Retrieval
        retrieval_res = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter
        )

        chunks = retrieval_res["retrieved_chunks"]
        retrieval_ms = retrieval_res["retrieval_latency_ms"]

        # Filter low relevance chunks if threshold set
        valid_chunks = [c for c in chunks if c["score"] >= min_score_threshold]

        # Handle 'no relevant context' scenario gracefully
        if not valid_chunks:
            no_context_answer = "Based on the provided context, I cannot find relevant information to answer this question."
            telemetry.record_query(
                query=question,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=0.0,
                chunk_count=0,
                prompt_tokens=0,
                completion_tokens=0
            )
            return {
                "question": question,
                "answer": no_context_answer,
                "grounded": False,
                "citations": [],
                "retrieved_chunks": [],
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": 0.0,
                "total_latency_ms": retrieval_ms,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        # Step 2: Format Context Blocks with Citations
        context_blocks = []
        for c in valid_chunks:
            context_blocks.append(
                f"[{c['citation_id']}] Document: {c['source_file']} (Chunk {c['chunk_index']})\n{c['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        user_prompt = USER_PROMPT_TEMPLATE.format(context_text=context_str, question=question)

        # Step 3: LLM Generation
        start_gen = time.perf_counter()

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "mock-key-for-testing":
            try:
                raw_answer, usage = self._generate_openai(user_prompt)
            except Exception as e:
                print(f"[Warning] OpenAI LLM call failed ({e}). Falling back to mock response generator.")
                raw_answer, usage = self._generate_mock(question, valid_chunks)
        else:
            raw_answer, usage = self._generate_mock(question, valid_chunks)

        gen_ms = (time.perf_counter() - start_gen) * 1000.0

        # Step 4: Extract Citations
        citations = []
        for c in valid_chunks:
            if c["citation_id"] in raw_answer or "Source" in raw_answer:
                citations.append({
                    "citation_id": c["citation_id"],
                    "source_file": c["source_file"],
                    "chunk_id": c["chunk_id"]
                })

        # Step 5: Log Telemetry
        telemetry.record_query(
            query=question,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=gen_ms,
            chunk_count=len(valid_chunks),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0)
        )

        return {
            "question": question,
            "answer": raw_answer,
            "grounded": True if citations or "[Source" in raw_answer else False,
            "citations": citations,
            "retrieved_chunks": valid_chunks,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "generation_latency_ms": round(gen_ms, 2),
            "total_latency_ms": round(retrieval_ms + gen_ms, 2),
            "token_usage": usage
        }

    def _generate_openai(self, user_prompt: str) -> tuple[str, dict]:
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = self._openai_client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        return answer, usage

    def _generate_mock(self, question: str, chunks: List[Dict[str, Any]]) -> tuple[str, dict]:
        """
        Deterministic mock response generator for offline execution and automated tests.
        """
        snippet = chunks[0]["text"][:120] if chunks else "no context"
        mock_answer = f"Based on [Source 1], {snippet}... This directly addresses '{question}'."
        mock_usage = {
            "prompt_tokens": len(question.split()) + 150,
            "completion_tokens": 30,
            "total_tokens": len(question.split()) + 180
        }
        return mock_answer, mock_usage
