# N2 · Clinical RAG: Retrieval Experiments and Cited Answers

Which chunking and retrieval setup actually finds the right passage when a patient asks a health question? This repo measures it instead of guessing, on a small corpus of NIH patient-education pages about diabetes, thyroid disease, and cholesterol. The best setup then feeds Claude, which answers only from the retrieved passages and cites the exact sentence behind each claim.

This is the second of a series of small experiments toward a clinical chart assistant ([N1: tool-calling agent](https://github.com/YaofuArea/n1-clinical-tool-calling)). The topics match the N1 synthetic EHR's labs: HbA1c, TSH, and LDL.

> The documents are public-domain pages from NIDDK and NHLBI (U.S. National Institutes of Health). This is a retrieval experiment, not medical advice.

## How it works

```
11 NIH pages ─▶ chunk (3 strategies) ─▶ retrieve (4 methods)
                                              │
question ─────────────────────────────────▶ top-k chunks ─▶ does any contain the evidence? ─▶ recall@k, MRR
```

| File | What it is |
|---|---|
| `fetch_docs.py` | Downloads 11 NIH pages as Markdown into `data/docs/`, keeping the source URL on the first line |
| `chunking.py` | 3 strategies: `fixed-300` and `fixed-800` (fixed character windows with overlap) and `section` (split at Markdown headings, long sections packed by paragraph, heading-only sections merged into the next one) |
| `retrieval.py` | 4 retrievers: BM25 keyword search (`rank-bm25`), dense embeddings (`BAAI/bge-small-en-v1.5`), hybrid (BM25 + embeddings with Reciprocal Rank Fusion), and rerank (embedding top 20 reordered by the cross-encoder `ms-marco-MiniLM-L-12-v2`). Models run locally via `fastembed` |
| `data/eval_questions.json` | 25 questions: 20 answerable, each with one or more exact evidence strings from the documents, and 5 the documents cannot answer |
| `evaluate.py` | Runs every chunking × retriever pair and reports recall@1/3/5 and MRR |
| `threshold.py` | Checks whether a top-1 score threshold separates answerable from unanswerable questions |
| `answer.py` | Full RAG: retrieves the top 3 chunks with the best setup and has Claude answer from them, with citations to the exact sentence and NIH page |

## Run

```bash
uv sync
uv run fetch_docs.py                     # already committed; only needed to refresh
uv run chunking.py                       # chunk counts and sizes per strategy
uv run retrieval.py "my thyroid is underactive and I feel tired"
uv run evaluate.py --misses              # results table + every miss (about 2 minutes)
uv run threshold.py                      # score-threshold refusal analysis

export ANTHROPIC_API_KEY=...             # only answer.py calls the Claude API
uv run answer.py "Can cholesterol pills raise my blood sugar?"
```

Everything except `answer.py` runs without an API key. The embedding model (about 67 MB) and the cross-encoder (about 120 MB) run on your machine and are downloaded on first run.

## How the evaluation works

Each answerable question lists one or more **evidence** strings copied word for word from the documents. Unanswerable questions have an empty list.

```json
{
  "question": "Is there a way to take insulin without shots?",
  "evidence": [
    "You may want to use an insulin inhaler to avoid using needles",
    "instead of using a needle to deliver the insulin"
  ],
  "type": "colloquial"
}
```

A question is a hit at k if any of the top-k chunks contains any of its evidence strings. Matching on text instead of chunk IDs means the same questions work for every chunking strategy.

- **R@k (recall@k):** share of questions whose evidence is in the top k.
- **MRR (mean reciprocal rank):** the average of 1/rank of the first hit (0 if not in the top 10). R@5 cannot tell rank 1 from rank 5; MRR can.

The questions come in four types:

| Type | Count | Example |
|---|---|---|
| keyword | 8 | "What does a high TSH level mean?" (uses the document's own terms) |
| colloquial | 10 | "Will cutting back on burgers and cheese help my bad cholesterol?" (the way a patient talks) |
| cross-topic | 2 | "Can cholesterol pills raise my blood sugar?" (answer is in a diabetes page) |
| unanswerable | 5 | "What is the normal range for TSH?" (the pages never give one) |

## Results

20 answerable questions. Best value per column in bold.

| Chunking | Retriever | R@1 | R@3 | R@5 | MRR | R@5 keyword | R@5 colloquial | R@5 cross-topic |
|---|---|---|---|---|---|---|---|---|
| fixed-300 | BM25 | 0.35 | 0.40 | 0.55 | 0.41 | 8/8 | 3/10 | 0/2 |
| fixed-300 | embedding | 0.65 | 0.80 | 0.80 | 0.73 | 8/8 | 7/10 | 1/2 |
| fixed-300 | hybrid | 0.50 | 0.70 | 0.70 | 0.60 | 8/8 | 6/10 | 0/2 |
| fixed-300 | rerank | 0.70 | 0.85 | 0.85 | 0.79 | 8/8 | 8/10 | 1/2 |
| fixed-800 | BM25 | 0.30 | 0.50 | 0.60 | 0.43 | 8/8 | 3/10 | 1/2 |
| fixed-800 | embedding | 0.65 | 0.75 | 0.75 | 0.72 | 7/8 | 7/10 | 1/2 |
| fixed-800 | hybrid | 0.45 | 0.75 | 0.80 | 0.62 | 8/8 | 6/10 | 2/2 |
| fixed-800 | rerank | 0.70 | 0.80 | 0.85 | 0.78 | 8/8 | 7/10 | 2/2 |
| section | BM25 | 0.40 | 0.65 | 0.70 | 0.51 | 8/8 | 6/10 | 0/2 |
| section | **embedding** | 0.75 | **1.00** | **1.00** | 0.87 | 8/8 | **10/10** | 2/2 |
| section | hybrid | 0.60 | 0.80 | 0.85 | 0.71 | 8/8 | 7/10 | 2/2 |
| section | **rerank** | **0.85** | 0.90 | 0.90 | **0.89** | 8/8 | 8/10 | 2/2 |

`answer.py` uses section + embedding with the top 3 chunks, where it finds the evidence for all 20 questions.

## What I learned

### Retrieval

- **Section chunking beats fixed windows for every retriever.** `fixed-800` chunks are larger than `section` chunks on average (777 vs 602 characters), yet they score worse. A section is about one topic, so its embedding is closer to a question about that topic. A fixed window often mixes two topics.
- **BM25 is fine on keywords and weak on how patients talk.** It found 8/8 keyword questions in every setup, but only 3–6 of 10 colloquial ones. "Insulin without shots" fails because the document says "needles".
- **Hybrid search made things worse here (negative result).** Reciprocal Rank Fusion gives both retrievers an equal vote. That helps when both are about equally good; here BM25 is much weaker, so its bad rankings pulled embedding's good ones down (section: MRR 0.87 → 0.71). Fusing two retrievers only pays off when the weaker one adds something the stronger one misses.
- **Reranking trades coverage for precision.** The cross-encoder puts the right chunk first more often (R@1 0.75 → 0.85, MRR 0.87 → 0.89) but pushes two answers out of the top 5. Both are colloquial symptom descriptions ("my thyroid is underactive and I feel tired", "I'm eating more than ever but still losing weight, and my heart races"); the reranker was trained on web search queries, not patient language. Since `answer.py` sends 3 chunks, coverage at 3 matters more than rank 1, so it keeps plain embeddings. With 20 questions, these one- or two-question gaps are within noise anyway.
- **Fixed windows cut evidence in half.** While writing the questions, two evidence strings could not be found with `fixed-300` at all, because a chunk boundary fell in the middle of the sentence (`...blood drawn for an A` / `ve to fast before having...`). Evidence no longer than the overlap (50 characters) always fits inside one chunk.

### The evaluation set needed evaluating too

The first version of the eval set had one evidence string per question. When reranking seemed to miss "Can cholesterol pills raise my blood sugar?", its top chunk turned out to answer the question too, from a cholesterol page instead of the diabetes page I had labeled. It was a false negative in the labels, not a retrieval miss.

To fix this fairly, I didn't only add that one answer, which would favor the method that found it. I collected the top chunks from all retrievers for every question, read them by hand, and added every passage that really answers the question: 11 more evidence strings for 9 of the 20 questions. A script checked that all 31 evidence strings appear in the documents.

| section | R@1 before → after | MRR before → after |
|---|---|---|
| embedding | 0.55 → 0.75 | 0.74 → 0.86 |
| rerank | 0.65 → 0.85 | 0.75 → 0.89 |

Every method went up, not just one, so the old labels were under-counting all of them. A missing label can quietly change which method looks best.

### Refusing to answer

**Score thresholds cannot tell answerable from unanswerable questions here (negative result).** The idea was to skip the Claude call when even the best chunk scores low. `threshold.py` shows the top-1 scores of the two groups overlap:

| section + … | Refuse ≥ 4 of 5 unanswerable | Answerable wrongly refused |
|---|---|---|
| embedding | top-1 ≤ 0.774 | 7 of 20 |
| rerank | top-1 ≤ −0.915 | 4 of 20 |

The wrongly refused questions are mostly colloquial ones like "I'm always thirsty and keep running to the bathroom". The deeper problem: a retrieval score measures whether a chunk is *on topic*, not whether it *contains the answer*. "What dose of levothyroxine should I take?" scores higher than 6 of the 20 answerable questions under reranking, because the pages talk about levothyroxine a lot; they just never give a dose. Only a reader of the text can notice that, so refusal is left to Claude (see experiments 2 and 3 below). Picking a threshold on these same 25 questions would also be tuning on the test set.

## Answer generation

`answer.py` sends each of the top 3 chunks (section + embedding) to Claude as a separate document with [citations](https://platform.claude.com/docs/en/build-with-claude/citations) enabled. The API returns the answer split into text blocks, and each claim carries the exact sentence it came from. The chunk then gives the NIH page URL.

| # | Question | What I expected | What happened |
|---|---|---|---|
| 1 | "Can cholesterol pills raise my blood sugar?" (cross-topic) | statins slightly raise diabetes risk, cited | Correct answer with 5 citations, all from the diabetes causes page, not the cholesterol pages. It also said what the documents *don't* cover (how much blood sugar changes day to day). |
| 2 | "What is the normal range for TSH?" (not in the corpus) | says the documents don't give a range | "The documents don't give a normal range for TSH." Claude knows the usual range, but it did not state a number the sources don't contain. It explained what high and low TSH mean instead, every sentence cited. |
| 3 | "How do I treat a broken arm?" (off-topic) | declines | Said the documents don't cover it and suggested seeing a professional. No citations, nothing invented. |

What I learned:

- **Grounding holds even when the model knows the answer.** Experiment 2 is the important one: refusing to give a TSH range it wasn't given is exactly the behavior a clinical tool needs.
- **Retrieval always returns k chunks, relevant or not.** For the broken-arm question it returned a hyperthyroidism chunk and two insulin chunks. Claude handled it; a score threshold would not have been reliable (see above).
- **Lists make citations coarse.** Citations are split at sentence boundaries, and Markdown list items have no periods, so citation [1] in experiment 1 is a whole medicine list plus the next sentence. The data format affects the answer layer too, not only retrieval.
- **Not every sentence is cited.** Claude added "Don't stop taking a prescribed medicine without talking to them first", which comes from the system prompt's safety rule, not the documents. A production system could flag uncited sentences.
- **Same question, different wording.** Running experiment 1 twice gave differently worded answers with the same 5 citations. Exact-match testing of answers would not work; tests should check the citations and facts instead.

## Data issues found

- **Title-only chunks (fixed).** `section` chunking used to turn a page title plus "On this page:" into its own chunk (e.g. `# Hypothyroidism (Underactive Thyroid) **On this page:**`). Such a chunk ranks high because the title matches the question, but it contains no answer; it was the top embedding result for both "my thyroid is underactive and I feel tired" and the unanswerable migraine question. Sections under 100 characters are now merged into the next section. This removed 7 chunks and raised section + embedding R@3 from 0.95 to 1.00 (one question, so a small gain).
- **Dropped words from the HTML.** Glossary links were lost during extraction, leaving gaps like "such as , or signs of" and "may prescribe , a medicine" in `cholesterol-diagnosis.md`. Questions about those words cannot be answered from this corpus.

## Limitations

- **20 answerable questions is small.** One question is 5 percentage points, so 0.85 vs 0.90 means nothing. Only large gaps (0.55 vs 1.00) are meaningful.
- **No separate test set.** All design choices (chunking, retriever, the 100-character merge rule) were compared on the same 20 questions they are reported on. The numbers are likely a little optimistic.
- **The questions were written with AI help** after reading the documents, which can favor phrasings that are easy to retrieve. Real user questions would be a better test.
- **Answers are checked by hand.** Retrieval is measured on all 20 questions, but answer quality was only checked on 3 example questions.

## Next

- Grade answers automatically (the RAG triad: is the retrieved context relevant, is the answer faithful to it, does it answer the question), including whether Claude refuses all 5 unanswerable questions.
- A larger reranker (e.g. `bge-reranker-base`) to see if the colloquial misses are a model-size problem.
- More chunking strategies: recursive splitting, and prefixing each chunk with its document title and heading.
