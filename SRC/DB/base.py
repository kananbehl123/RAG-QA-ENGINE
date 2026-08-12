from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    """
    Abstract Interface for Vector Database implementations.
    Guarantees consistent operations across LanceDB, ChromaDB, etc.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Store vector chunks with metadata. Returns number of newly inserted vectors.
        Must enforce idempotency (no duplicate vectors based on chunk_hash).
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most similar vector chunks matching optional metadata filter.
        Returns list of dictionary results with 'text', 'metadata', and 'score' / 'distance'.
        """
        pass

    @abstractmethod
    def get_existing_hashes(self) -> set:
        """
        Return set of all chunk_hashes currently present in the store.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Return total vector count in the database.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all records in the vector database.
        """
        pass
