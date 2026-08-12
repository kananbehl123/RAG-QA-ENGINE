import hashlib
import numpy as np
from typing import List
from src.config import settings


class EmbeddingManager:
    """
    Embedding Provider supporting OpenAI API, SentenceTransformers, and Mock fallback embeddings.
    Records embedding model name and dimensionality.
    """

    def __init__(self, provider: str = None, model_name: str = None, dimension: int = None):
        self.provider = provider or settings.EMBEDDING_PROVIDER
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIM
        self._openai_client = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.provider == "openai" and settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "mock-key-for-testing":
            try:
                return self._embed_openai(texts)
            except Exception as e:
                print(f"[Warning] OpenAI embedding failed ({e}). Falling back to mock embeddings.")
                return self._embed_mock(texts)
        elif self.provider == "sentence-transformers":
            try:
                return self._embed_sentence_transformers(texts)
            except Exception as e:
                print(f"[Warning] SentenceTransformers failed ({e}). Falling back to mock embeddings.")
                return self._embed_mock(texts)
        else:
            return self._embed_mock(texts)

    def embed_query(self, query: str) -> List[float]:
        results = self.embed_documents([query])
        return results[0] if results else [0.0] * self.dimension

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = self._openai_client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [data.embedding for data in response.data]

    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def _embed_mock(self, texts: List[str]) -> List[List[float]]:
        """
        Deterministic mock vector generator for unit tests and local offline evaluation.
        Uses SHA-256 seed to map text content to a normalized vector of fixed dimensionality.
        """
        embeddings = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dimension)
            # Normalize vector to unit L2 length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings
