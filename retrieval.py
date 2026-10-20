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


RETRIEVERS = {"bm25": BM25Retriever, "embedding": EmbeddingRetriever}


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "What is a normal A1C level?"
    chunks = STRATEGIES["section"]()
    print(f"Q: {query}   (chunking: section, {len(chunks)} chunks)")
    for name, retriever_class in RETRIEVERS.items():
        retriever = retriever_class(chunks)
        print(f"\n== {name}")
        for chunk, score in retriever.search(query, k=3):
            preview = chunk.text.replace("\n", " ")[:110]
            print(f"  {score:6.2f}  {chunk.chunk_id:28} {preview}")
