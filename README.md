# N2 · Clinical RAG Retrieval Experiments

Which chunking and retrieval setup actually finds the right passage when a patient asks a health question? This repo measures it instead of guessing, on a small corpus of NIH patient-education pages about diabetes, thyroid disease, and cholesterol.

This is the second of a series of small experiments toward a clinical chart assistant ([N1: tool-calling agent](https://github.com/YaofuArea/n1-clinical-tool-calling)). The topics match the N1 synthetic EHR's labs: HbA1c, TSH, and LDL.

> The documents are public-domain pages from NIDDK and NHLBI (U.S. National Institutes of Health). This is a retrieval experiment, not medical advice.

## How it works

```
11 NIH pages ─▶ chunk (3 strategies) ─▶ index (BM25 or embeddings)
                                              │
question ─────────────────────────────────▶ top-k chunks ─▶ does any contain the evidence? ─▶ recall@k
```

| File | What it is |
|---|---|
| `fetch_docs.py` | Downloads 11 NIH pages as Markdown into `data/docs/`, keeping the source URL on the first line |
| `chunking.py` | 3 strategies: `fixed-300` and `fixed-800` (fixed character windows with overlap) and `section` (split at Markdown headings, long sections packed by paragraph) |
| `retrieval.py` | 2 retrievers: BM25 keyword search (`rank-bm25`) and dense embeddings (`BAAI/bge-small-en-v1.5` via `fastembed`, runs locally) |
| `data/eval_questions.json` | 20 questions, each with an exact evidence substring from the documents |
| `evaluate.py` | Runs every chunking × retriever pair and reports recall@1/3/5 |

## Run

```bash
uv sync
uv run fetch_docs.py                     # already committed; only needed to refresh
uv run chunking.py                       # chunk counts and sizes per strategy
uv run retrieval.py "my thyroid is underactive and I feel tired"
uv run evaluate.py --misses              # results table + every miss
```

No API key is needed: the embedding model runs on your machine (about 67 MB, downloaded on first run).

## How the evaluation works

Each question has an **evidence** string copied word for word from one document, e.g.

```json
{
  "question": "Is there a way to take insulin without shots?",
  "evidence": "You may want to use an insulin inhaler to avoid using needles",
  "doc_id": "diabetes-medicines",
  "type": "colloquial"
}
```

A question is a hit at k if any of the top-k chunks contains the evidence. Matching on text instead of chunk IDs means the same questions work for every chunking strategy.

The 20 questions come in three types:

| Type | Count | Example |
|---|---|---|
| keyword | 8 | "What does a high TSH level mean?" (uses the document's own terms) |
| colloquial | 10 | "Will cutting back on burgers and cheese help my bad cholesterol?" (the way a patient talks) |
| cross-topic | 2 | "Can cholesterol pills raise my blood sugar?" (answer is in a diabetes page) |

## Results

| Chunking | Retriever | R@1 | R@3 | R@5 | R@5 keyword | R@5 colloquial | R@5 cross-topic |
|---|---|---|---|---|---|---|---|
| fixed-300 | BM25 | 0.30 | 0.35 | 0.45 | 7/8 | 2/10 | 0/2 |
| fixed-300 | embedding | 0.50 | 0.70 | 0.80 | 8/8 | 7/10 | 1/2 |
| fixed-800 | BM25 | 0.20 | 0.50 | 0.60 | 8/8 | 3/10 | 1/2 |
| fixed-800 | embedding | 0.40 | 0.65 | 0.70 | 7/8 | 6/10 | 1/2 |
| section | BM25 | 0.30 | 0.45 | 0.65 | 8/8 | 5/10 | 0/2 |
| **section** | **embedding** | **0.55** | **0.90** | **1.00** | **8/8** | **10/10** | **2/2** |

## What I learned

- **Section + embedding wins clearly.** It found the evidence in the top 5 for all 20 questions, and in the top 3 for 18. Every other setup missed at least 4.
- **BM25 is fine on keywords and weak on how patients talk.** On keyword questions BM25 matched embeddings (7–8 of 8). On colloquial questions it found only 2–5 of 10. "Insulin without shots" fails because the document says "needles". In a question like "my thyroid is underactive and I feel tired", common words such as "my" and "I" pulled a diabetes chunk to BM25's rank 1.
- **Structure matters more than size.** `fixed-800` chunks are larger than `section` chunks (avg 777 vs 581 characters), so they hold more text, yet they score worse. A section is about one topic, so its embedding is closer to a question about that topic. A fixed window often mixes two topics.
- **Fixed windows cut evidence in half.** While writing the questions, two evidence strings could not be found with `fixed-300` at all, because a chunk boundary fell in the middle of the sentence (`...blood drawn for an A` / `ve to fast before having...`). Evidence no longer than the overlap (50 characters) always fits inside one chunk. Longer overlap means fewer cut sentences, at the cost of more chunks.

## Data issues found

- **Title-only chunks.** `section` chunking turns a page title plus "On this page:" into its own chunk (e.g. `hypothyroidism#0` is just `# Hypothyroidism (Underactive Thyroid) **On this page:**`). It ranks high because the title matches the question, but it contains no answer.
- **Dropped words from the HTML.** Glossary links were lost during extraction, leaving gaps like "such as , or signs of" and "may prescribe , a medicine" in `cholesterol-diagnosis.md`. Questions about those words cannot be answered from this corpus.

## Limitations

- **20 questions is small.** One question is 5 percentage points, so 0.65 vs 0.70 means nothing. Only large gaps (0.45 vs 1.00) are meaningful.
- **The questions were written with AI help** after reading the documents, which can favor phrasings that are easy to retrieve. Real user questions would be a better test.
- **Retrieval only.** This measures whether the right passage is found, not whether a final answer built from it is correct.

## Next

- Generate answers with Claude from the top section + embedding chunks, citing each chunk's source URL.
- Try more chunking strategies: recursive splitting, and prefixing each chunk with its document title and heading.
- Hybrid search (BM25 + embeddings) to keep keyword precision without losing colloquial questions.
