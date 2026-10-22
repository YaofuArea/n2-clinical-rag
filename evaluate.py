"""Measure retrieval quality: recall@k for every chunking strategy x retriever.

A question counts as a hit at k if any of the top-k chunks contains the
question's evidence text. This checks retrieval only; no answer is generated.

Usage: uv run evaluate.py          (summary table)
       uv run evaluate.py --misses (also list questions missed at k=5)
"""

import json
import sys
from pathlib import Path

from chunking import STRATEGIES
from retrieval import RETRIEVERS

QUESTIONS = json.loads(
    (Path(__file__).parent / "data" / "eval_questions.json").read_text()
)
KS = [1, 3, 5]


def first_hit_rank(results, evidence: str) -> int | None:
    """1-based rank of the first chunk that contains the evidence, or None."""
    for rank, (chunk, _score) in enumerate(results, start=1):
        if evidence in chunk.text:
            return rank
    return None


def main() -> None:
    show_misses = "--misses" in sys.argv
    types = sorted({q["type"] for q in QUESTIONS})

    header = f"{'chunking':10} {'retriever':10} " + " ".join(f"R@{k:<3}" for k in KS)
    header += "   R@5 by type: " + "  ".join(types)
    print(f"{len(QUESTIONS)} questions\n\n{header}\n{'-' * len(header)}")

    misses = []
    for strategy_name, make_chunks in STRATEGIES.items():
        chunks = make_chunks()
        for retriever_name, retriever_class in RETRIEVERS.items():
            retriever = retriever_class(chunks)
            ranks = []
            for q in QUESTIONS:
                rank = first_hit_rank(
                    retriever.search(q["question"], k=max(KS)), q["evidence"]
                )
                ranks.append(rank)
                if rank is None:
                    misses.append((strategy_name, retriever_name, q))

            recalls = [
                sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS
            ]
            by_type = []
            for t in types:
                typed = [r for r, q in zip(ranks, QUESTIONS) if q["type"] == t]
                by_type.append(f"{sum(r is not None for r in typed)}/{len(typed)}")
            print(
                f"{strategy_name:10} {retriever_name:10} "
                + " ".join(f"{r:5.2f}" for r in recalls)
                + "   "
                + "  ".join(f"{b:>{len(t)}}" for b, t in zip(by_type, types))
            )

    if show_misses:
        print("\nMissed at k=5:")
        for strategy_name, retriever_name, q in misses:
            print(
                f"  {strategy_name:10} {retriever_name:10} [{q['type']}] {q['question']}"
            )


if __name__ == "__main__":
    main()
