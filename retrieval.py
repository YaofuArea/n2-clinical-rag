"""Two ways to find the chunks most relevant to a question.

- BM25: keyword matching. Rewards chunks that contain the question's words,
  especially rare words.
- Embedding: semantic matching. A local model turns text into vectors and we
  rank chunks by cosine similarity to the question.

Usage: uv run retrieval.py "What is a normal A1C level?"
"""

import re
import sys

import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

from chunking import STRATEGIES, Chunk

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # small (~67 MB), runs locally


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.index.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class EmbeddingRetriever:
    _model = None  # loaded once and shared, since loading takes a few seconds

    def __init__(self, chunks: list[Chunk]):
        if EmbeddingRetriever._model is None:
            EmbeddingRetriever._model = TextEmbedding(EMBEDDING_MODEL)
        self.chunks = chunks
        # The model returns unit-length vectors, so a dot product is the cosine similarity.
        self.vectors = np.array(
            list(self._model.passage_embed([c.text for c in chunks]))
        )

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        query_vector = next(iter(self._model.query_embed(query)))
        scores = self.vectors @ query_vector
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class HybridRetriever:
    """BM25 + embeddings, merged with Reciprocal Rank Fusion (RRF).

    BM25 scores (0 to ~10) and cosine similarities (0 to 1) are on different
    scales, so RRF ignores the scores and uses only each chunk's rank:
        score(chunk) = sum over retrievers of 1 / (RRF_K + rank)
    A chunk ranked high by both retrievers beats one ranked high by only one.
    """

    RRF_K = 60  # the usual constant; larger values flatten the gap between ranks
    CANDIDATES = 50  # how deep into each retriever's ranking we look

    def __init__(self, chunks: list[Chunk]):
        self.retrievers = [BM25Retriever(chunks), EmbeddingRetriever(chunks)]

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in self.retrievers:
            results = retriever.search(query, k=self.CANDIDATES)
            for rank, (chunk, _score) in enumerate(results, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0) + 1 / (
                    self.RRF_K + rank
                )
                by_id[chunk.chunk_id] = chunk
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(by_id[chunk_id], fused[chunk_id]) for chunk_id in top]


RETRIEVERS = {
    "bm25": BM25Retriever,
    "embedding": EmbeddingRetriever,
    "hybrid": HybridRetriever,
}


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is a normal A1C level?"
    chunks = STRATEGIES["section"]()
    print(f"Q: {query}   (chunking: section, {len(chunks)} chunks)")
    for name, retriever_class in RETRIEVERS.items():
        retriever = retriever_class(chunks)
        print(f"\n== {name}")
        for chunk, score in retriever.search(query, k=3):
            preview = chunk.text.replace("\n", " ")[:110]
            print(f"  {score:7.4f}  {chunk.chunk_id:28} {preview}")
