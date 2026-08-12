import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.db.base import BaseVectorStore


class ChromaDBStore(BaseVectorStore):
    """
    ChromaDB Secondary Benchmark Vector Database Store.
    Used for evaluation baseline vs LanceDB.
    """

    COLLECTION_NAME = "document_chunks"

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = None
        self.collection = None
        self._init_db()

    def _init_db(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            pass

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return 0

        existing_hashes = self.get_existing_hashes()
        ids = []
        vectors = []
        documents = []
        metadatas = []

        for chunk, vector in zip(chunks, embeddings):
            c_hash = chunk.get("chunk_hash", "")
            if c_hash and c_hash in existing_hashes:
                # Idempotent skip
                continue

            ids.append(chunk["chunk_id"])
            vectors.append(vector)
            documents.append(chunk["text"])
            metadatas.append({
                "doc_id": str(chunk["doc_id"]),
                "source_file": str(chunk["source_file"]),
                "file_type": str(chunk.get("file_type", "")),
                "chunk_index": int(chunk["chunk_index"]),
                "chunk_hash": str(c_hash),
                "created_at": int(chunk.get("created_at", 0))
            })

        if not ids:
            return 0

        if self.collection is None:
            self._init_db()

        if self.collection:
            self.collection.add(
                ids=ids,
                embeddings=vectors,
                documents=documents,
                metadatas=metadatas
            )

        return len(ids)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.collection is None:
            return []

        where_clause = metadata_filter if metadata_filter else None
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_clause
        )

        formatted = []
        if res and res.get("ids") and res["ids"][0]:
            ids = res["ids"][0]
            docs = res["documents"][0]
            metas = res["metadatas"][0]
            dists = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(ids)

            for c_id, doc, meta, dist in zip(ids, docs, metas, dists):
                score = 1.0 - dist if dist <= 1.0 else 1.0 / (1.0 + dist)
                formatted.append({
                    "chunk_id": c_id,
                    "doc_id": meta.get("doc_id"),
                    "source_file": meta.get("source_file"),
                    "file_type": meta.get("file_type"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": doc,
                    "chunk_hash": meta.get("chunk_hash"),
                    "distance": dist,
                    "score": score
                })
        return formatted

    def get_existing_hashes(self) -> set:
        if self.collection is None:
            return set()
        try:
            data = self.collection.get(include=["metadatas"])
            if data and data.get("metadatas"):
                return {m.get("chunk_hash") for m in data["metadatas"] if m and "chunk_hash" in m}
        except Exception:
            pass
        return set()

    def count(self) -> int:
        if self.collection is None:
            return 0
        try:
            return self.collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        if self.client:
            try:
                self.client.delete_collection(self.COLLECTION_NAME)
                self.collection = self.client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
