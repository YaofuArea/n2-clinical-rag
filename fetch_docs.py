"""Download a small corpus of public NIH patient-education pages as Markdown.

Content from NIDDK and NHLBI is published by the U.S. federal government
and is in the public domain.

Usage: uv run fetch_docs.py
"""

from pathlib import Path

import trafilatura

DOCS = {
    # Diabetes (HbA1c)
    "diabetes-tests-diagnosis": "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis",
    "a1c-test": "https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test",
    "diabetes-symptoms-causes": "https://www.niddk.nih.gov/health-information/diabetes/overview/symptoms-causes",
    "managing-diabetes": "https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes",
    "diabetes-medicines": "https://www.niddk.nih.gov/health-information/diabetes/overview/insulin-medicines-treatments",
    # Thyroid (TSH)
    "hypothyroidism": "https://www.niddk.nih.gov/health-information/endocrine-diseases/hypothyroidism",
    "hyperthyroidism": "https://www.niddk.nih.gov/health-information/endocrine-diseases/hyperthyroidism",
    "thyroid-tests": "https://www.niddk.nih.gov/health-information/diagnostic-tests/thyroid",
    # Cholesterol (LDL)
    "cholesterol-causes": "https://www.nhlbi.nih.gov/health/blood-cholesterol/causes",
    "cholesterol-diagnosis": "https://www.nhlbi.nih.gov/health/blood-cholesterol/diagnosis",
    "cholesterol-treatment": "https://www.nhlbi.nih.gov/health/blood-cholesterol/treatment",
}

OUT_DIR = Path(__file__).parent / "data" / "docs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for doc_id, url in DOCS.items():
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html, output_format="markdown") if html else None
        if not text:
            print(f"FAILED  {doc_id}")
            continue
        # Keep the source URL in the file so every chunk can be cited later.
        (OUT_DIR / f"{doc_id}.md").write_text(f"Source: {url}\n\n{text}\n")
        print(f"ok      {doc_id:28} {len(text):>6} chars")


if __name__ == "__main__":
    main()
