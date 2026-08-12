from pathlib import Path
from typing import List, Dict, Any, Optional

from src.db.base import BaseVectorStore


class LanceDBStore(BaseVectorStore):
    """
    LanceDB Primary Embedded Vector Database Store.
    Zero-pod, disk-backed, high-performance columnar vector store.
    """

    TABLE_NAME = "document_chunks"

    def __init__(self, db_path: str):
        self.db_path = db_path

        Path(db_path).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = None
        self.table = None

        self._init_db()

    def _init_db(self):
        try:
            import lancedb

            self.client = lancedb.connect(self.db_path)

            # Try to open existing table
            try:
                self.table = self.client.open_table(self.TABLE_NAME)
            except Exception:
                # Table does not exist yet
                self.table = None

        except Exception:
            self.client = None
            self.table = None

    def _get_tables(self) -> List[str]:
        if self.client is None:
            return []

        try:
            if hasattr(self.client, "list_tables"):
                result = self.client.list_tables()

                # Newer LanceDB versions
                if hasattr(result, "tables"):
                    return result.tables

                # Older versions
                return result

            elif hasattr(self.client, "table_names"):
                return self.client.table_names()

        except Exception:
            return []

        return []

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:

        if (
            not chunks
            or not embeddings
            or len(chunks) != len(embeddings)
        ):
            return 0

        existing_hashes = self.get_existing_hashes()
        new_records = []

        for chunk, vector in zip(chunks, embeddings):

            c_hash = chunk.get("chunk_hash", "")

            # Idempotent skip: duplicate vector chunk
            if c_hash and c_hash in existing_hashes:
                continue

            record = {
                "vector": vector,
                "id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source_file": chunk["source_file"],
                "file_type": chunk.get("file_type", ""),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "chunk_hash": c_hash,
                "created_at": chunk.get("created_at", 0),
            }

            new_records.append(record)

        if not new_records:
            return 0

        import lancedb

        if self.client is None:
            self.client = lancedb.connect(self.db_path)

        # If table is already open, simply add records
        if self.table is not None:
            self.table.add(new_records)
            return len(new_records)

        # Otherwise try to open the existing table
        try:
            self.table = self.client.open_table(self.TABLE_NAME)

            self.table.add(new_records)

        except Exception:

            # Table does not exist yet, so create it
            self.table = self.client.create_table(
                self.TABLE_NAME,
                data=new_records
            )

        return len(new_records)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        if self.table is None:
            return []

        search_query = (
            self.table
            .search(query_vector)
            .limit(top_k)
        )

        # Apply metadata filtering if specified
        if metadata_filter:

            conditions = []

            for key, value in metadata_filter.items():

                if isinstance(value, str):
                    conditions.append(
                        f"{key} = '{value}'"
                    )
                else:
                    conditions.append(
                        f"{key} = {value}"
                    )

            if conditions:
                where_clause = " AND ".join(conditions)
                search_query = search_query.where(
                    where_clause
                )

        results = search_query.to_list()

        formatted = []

        for result in results:

            distance = result.get(
                "_distance",
                0.0
            )

            score = 1.0 / (1.0 + distance)

            formatted.append({
                "chunk_id": result.get("id"),
                "doc_id": result.get("doc_id"),
                "source_file": result.get("source_file"),
                "file_type": result.get("file_type"),
                "chunk_index": result.get("chunk_index"),
                "text": result.get("text"),
                "chunk_hash": result.get("chunk_hash"),
                "distance": distance,
                "score": score,
            })

        return formatted

    def get_existing_hashes(self) -> set:

        if self.table is None:
            return set()

        try:

            arrow_table = self.table.to_arrow()

            if "chunk_hash" in arrow_table.column_names:

                hashes = arrow_table[
                    "chunk_hash"
                ].to_pylist()

                return set(
                    h for h in hashes
                    if h
                )

        except Exception:
            pass

        return set()

    def count(self) -> int:

        if self.table is None:
            return 0

        try:
            return len(self.table)

        except Exception:
            return 0

    def clear(self) -> None:

        if self.client is None:
            return

        try:

            tables = self._get_tables()

            if self.TABLE_NAME in tables:

                self.client.drop_table(
                    self.TABLE_NAME
                )

            self.table = None

        except Exception:
            self.table = None
            