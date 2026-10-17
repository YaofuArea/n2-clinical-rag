"""Split the documents in data/docs into chunks, using different strategies.

Every chunk remembers which document and which heading it came from,
so an answer built from it can cite its source.

Usage: uv run chunking.py      (prints stats for each strategy)
"""

from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "data" / "docs"


@dataclass
class Chunk:
    chunk_id: str  # e.g. "a1c-test#3"
    doc_id: str  # file name without .md
    source: str  # original URL
    heading: str  # nearest heading above the chunk
    text: str


def load_docs() -> list[tuple[str, str, str]]:
    """Return (doc_id, source_url, body) for every document."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        first_line, body = path.read_text().split("\n", 1)
        docs.append((path.stem, first_line.removeprefix("Source: "), body.strip()))
    return docs


def _heading_at(body: str, pos: int) -> str:
    """The last Markdown heading that starts before character `pos`."""
    heading = ""
    offset = 0
    for line in body.split("\n"):
        if offset > pos:
            break
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
        offset += len(line) + 1
    return heading


def chunk_fixed(size: int, overlap: int) -> list[Chunk]:
    """Cut every `size` characters; consecutive chunks share `overlap` characters.

    Simple and uniform, but it can cut a sentence (or a word) in half.
    """
    chunks = []
    for doc_id, source, body in load_docs():
        step = size - overlap
        for i, start in enumerate(range(0, len(body), step)):
            text = body[start : start + size]
            chunks.append(
                Chunk(f"{doc_id}#{i}", doc_id, source, _heading_at(body, start), text)
            )
            if start + size >= len(body):
                break
    return chunks


def chunk_by_section(max_chars: int) -> list[Chunk]:
    """One chunk per Markdown section; split long sections at paragraph breaks.

    Each chunk is one topic, but chunk sizes vary a lot.
    """
    chunks = []
    for doc_id, source, body in load_docs():
        # Group lines into sections, starting a new one at every heading.
        sections: list[tuple[str, list[str]]] = [("", [])]
        for line in body.split("\n"):
            if line.startswith("#"):
                sections.append((line.lstrip("#").strip(), [line]))
            else:
                sections[-1][1].append(line)

        i = 0
        for heading, lines in sections:
            text = "\n".join(lines).strip()
            if not text:
                continue
            # Pack paragraphs into pieces no longer than max_chars.
            pieces, current = [], ""
            for para in text.split("\n\n"):
                if current and len(current) + len(para) + 2 > max_chars:
                    pieces.append(current)
                    current = para
                else:
                    current = f"{current}\n\n{para}" if current else para
            pieces.append(current)
            for piece in pieces:
                chunks.append(Chunk(f"{doc_id}#{i}", doc_id, source, heading, piece))
                i += 1
    return chunks


STRATEGIES = {
    "fixed-300": lambda: chunk_fixed(size=300, overlap=50),
    "fixed-800": lambda: chunk_fixed(size=800, overlap=100),
    "section": lambda: chunk_by_section(max_chars=1500),
}


if __name__ == "__main__":
    for name, make in STRATEGIES.items():
        chunks = make()
        lengths = [len(c.text) for c in chunks]
        print(
            f"{name:10} {len(chunks):4} chunks   "
            f"avg {sum(lengths) // len(lengths):4}  min {min(lengths):4}  max {max(lengths):4} chars"
        )
    example = STRATEGIES["fixed-300"]()[5]
    print(
        f"\nExample fixed-300 chunk {example.chunk_id} (heading: {example.heading!r}):"
    )
    print(repr(example.text))
