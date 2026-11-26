"""Persistent vector store backed by Chroma.

This is the only file that talks to the vector database. Swap Chroma for
FAISS/pgvector here later without touching the rest of the RAG package.

Uses cosine distance so query() can return an interpretable similarity in
[0, 1] (1 = identical meaning), which the retriever thresholds on.
NOTE: changing the distance space requires a fresh index — delete rag/store
and re-run ingest after upgrading to this version.
"""
from __future__ import annotations

from .embed import embed


class VectorStore:
    def __init__(
        self,
        path: str = "rag/store",
        collection: str = "agentigrid_kb",
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
    ):
        import chromadb  # imported lazily so the package loads even without chromadb

        self._client = chromadb.PersistentClient(path=path)
        # cosine space -> distance in [0, 2]; similarity = 1 - distance
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )
        self._host, self._model = host, model

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._col.add(
            ids=[doc_id],
            embeddings=[embed(text, self._host, self._model)],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, k: int = 5) -> list[tuple[str, dict, float]]:
        """Return [(document, metadata, similarity), ...], best first.

        similarity = 1 - cosine_distance, so ~1.0 is a very close match and
        values near 0 (or below) are weak/unrelated.
        """
        res = self._col.query(
            query_embeddings=[embed(text, self._host, self._model)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append((doc, meta, 1.0 - float(dist)))
        return out

    def count(self) -> int:
        return self._col.count()

    def all_documents(self) -> list[tuple[str, str, dict]]:
        """Every stored chunk as (id, document, metadata) -- for offline tools
        such as relevance labelling; not used on the retrieval path."""
        res = self._col.get(include=["documents", "metadatas"])
        return list(zip(res.get("ids", []), res.get("documents", []), res.get("metadatas", [])))

    def reset(self) -> None:
        """Drop and recreate the collection (a fresh index, no stale chunks)."""
        name = self._col.name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
