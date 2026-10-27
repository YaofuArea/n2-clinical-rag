"""Answer a health question from the retrieved chunks, with citations.

Retrieval uses the best setup from evaluate.py (section chunks + embeddings).
Each retrieved chunk is sent to Claude as a separate document with citations
enabled, so every claim in the answer points back to the exact sentence it
came from, and to the NIH page that sentence is on.

Usage: uv run answer.py "Can cholesterol pills raise my blood sugar?"
"""

import sys

import anthropic

from chunking import STRATEGIES
from retrieval import EmbeddingRetriever

MODEL = "claude-opus-5-5"
TOP_K = 3  # section + embedding found the evidence in the top 3 for 18 of 20 questions

SYSTEM = """You answer patient health questions using only the documents provided.
- Use only facts stated in the documents. Do not add facts from your own knowledge.
- If the documents do not answer the question, say so plainly instead of guessing.
- Answer in plain language, in a few sentences.
- You give general information, not medical advice; for a personal decision, suggest asking a health care professional."""

client = anthropic.Anthropic()
retriever = EmbeddingRetriever(STRATEGIES["section"]())


def answer(question: str) -> None:
    chunks = [chunk for chunk, _score in retriever.search(question, k=TOP_K)]

    # One document per chunk: the title says which page and section it is from,
    # and the context carries the source URL so we can show it with the citation.
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c.text},
            "title": f"{c.doc_id} / {c.heading}",
            "context": f"Source: {c.source}",
            "citations": {"enabled": True},
        }
        for c in chunks
    ]

    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [*documents, {"type": "text", "text": question}],
            }
        ],
        output_config={"effort": "medium"},
        # If a safety classifier declines the request, the API retries it on a
        # recommended fallback model inside the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":
        print("[declined by safety classifier]")
        return

    # With citations on, the answer comes back as several text blocks; the ones
    # that make a claim carry a list of citations. Number each cited sentence once.
    cited: dict[tuple[int, int], int] = {}  # (document_index, start_char) -> [n]
    sources = []
    answer_text = ""
    for block in response.content:
        if block.type != "text":
            continue
        answer_text += block.text
        for citation in block.citations or []:
            key = (citation.document_index, citation.start_char_index)
            if key not in cited:
                cited[key] = len(cited) + 1
                sources.append(
                    (cited[key], chunks[citation.document_index], citation.cited_text)
                )
            answer_text += f" [{cited[key]}]"

    print(f"Q: {question}\n")
    print(f"A: {answer_text.strip()}\n")
    print("Retrieved: " + ", ".join(c.chunk_id for c in chunks))
    for n, chunk, cited_text in sources:
        quote = " ".join(cited_text.split())
        print(f'  [{n}] {chunk.chunk_id}  {chunk.source}\n      "{quote}"')


if __name__ == "__main__":
    answer(" ".join(sys.argv[1:]) or "Can cholesterol pills raise my blood sugar?")
