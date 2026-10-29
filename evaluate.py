"""Measure retrieval quality for every chunking strategy x retriever.

A question counts as a hit at k if any of the top-k chunks contains the
question's evidence text. This checks retrieval only; no answer is generated.

Metrics (answerable questions only):
- R@k: share of questions whose evidence is in the top k.
- MRR: mean of 1/rank of the first hit (0 if not in the top 10). Unlike R@5,
  it still moves when a hit climbs from rank 3 to rank 1.

Unanswerable questions (evidence: null) have no correct chunk, so they are
left out here; they are used to test refusing to answer.

Usage: uv run evaluate.py          (summary table)
       uv run evaluate.py --misses (also list questions missed at k=5)
"""

import json
import sys
from pathlib import Path

from chunking import STRATEGIES
from retrieval import RETRIEVERS

ALL_QUESTIONS = json.loads(
    (Path(__file__).parent / "data" / "eval_questions.json").read_text()
)
QUESTIONS = [q for q in ALL_QUESTIONS if q["evidence"] is not None]
KS = [1, 3, 5]
MRR_DEPTH = 10  # a hit below rank 10 counts as a miss for MRR


def first_hit_rank(results, evidence: str) -> int | None:
    """1-based rank of the first chunk that contains the evidence, or None."""
    for rank, (chunk, _score) in enumerate(results, start=1):
        if evidence in chunk.text:
            return rank
    return None


def main() -> None:
    show_misses = "--misses" in sys.argv
    types = sorted({q["type"] for q in QUESTIONS})

    header = (
        f"{'chunking':10} {'retriever':10} "
        + " ".join(f"R@{k:<3}" for k in KS)
        + "  MRR "
        + "   R@5 by type: "
        + "  ".join(types)
    )
    skipped = len(ALL_QUESTIONS) - len(QUESTIONS)
    print(f"{len(QUESTIONS)} answerable questions ({skipped} unanswerable skipped)")
    print(f"\n{header}\n{'-' * len(header)}")

    misses = []
    for strategy_name, make_chunks in STRATEGIES.items():
        chunks = make_chunks()
        for retriever_name, retriever_class in RETRIEVERS.items():
            retriever = retriever_class(chunks)
            ranks = []
            for q in QUESTIONS:
                results = retriever.search(q["question"], k=MRR_DEPTH)
                rank = first_hit_rank(results, q["evidence"])
                ranks.append(rank)
                if rank is None or rank > max(KS):
                    misses.append((strategy_name, retriever_name, q))

            recalls = [
                sum(r is not None and r <= k for r in ranks) / len(ranks) for k in KS
            ]
            mrr = sum(1 / r for r in ranks if r is not None) / len(ranks)
            by_type = []
            for t in types:
                typed = [r for r, q in zip(ranks, QUESTIONS) if q["type"] == t]
                hits = sum(r is not None and r <= max(KS) for r in typed)
                by_type.append(f"{hits}/{len(typed)}")
            print(
                f"{strategy_name:10} {retriever_name:10} "
                + " ".join(f"{r:5.2f}" for r in recalls)
                + f" {mrr:5.2f}"
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
