"""Can a score threshold tell answerable questions from unanswerable ones?

The idea: if even the best chunk scores low, nothing relevant was found, so
skip the Claude call and refuse. For each retriever this prints every
question's top-1 score, then, for each possible threshold, how many
unanswerable questions it would refuse and how many answerable ones it would
wrongly refuse.

Usage: uv run threshold.py
"""

import json
from pathlib import Path

from chunking import STRATEGIES
from retrieval import EmbeddingRetriever, RerankRetriever

QUESTIONS = json.loads(
    (Path(__file__).parent / "data" / "eval_questions.json").read_text()
)


def main() -> None:
    chunks = STRATEGIES["section"]()
    for name, retriever_class in [
        ("embedding", EmbeddingRetriever),
        ("rerank", RerankRetriever),
    ]:
        retriever = retriever_class(chunks)
        rows = []  # (top-1 score, answerable?, question)
        for q in QUESTIONS:
            _chunk, score = retriever.search(q["question"], k=1)[0]
            rows.append((score, bool(q["evidence"]), q["question"]))
        rows.sort()

        n_ans = sum(a for _, a, _ in rows)
        n_unans = len(rows) - n_ans
        print(f"\n== section + {name}: top-1 score, lowest first")
        for score, answerable, question in rows:
            label = "answerable  " if answerable else "UNANSWERABLE"
            print(f"  {score:8.4f}  {label}  {question}")

        # Refuse when the top-1 score is at or below t. Try each score as t.
        print("\n  refuse if top-1 <= t   unanswerable refused   answerable refused")
        for t, _a, _q in rows:
            refused_unans = sum(s <= t and not a for s, a, _ in rows)
            refused_ans = sum(s <= t and a for s, a, _ in rows)
            print(
                f"  {t:19.4f}   {refused_unans:>12}/{n_unans}"
                f"   {refused_ans:>16}/{n_ans}"
            )


if __name__ == "__main__":
    main()
