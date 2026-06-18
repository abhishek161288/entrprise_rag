# app/retrieval/bm25_index.py

from pathlib import Path
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

documents = []

for file in KNOWLEDGE_BASE_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append(
        {
            "source": file.name,
            "text": text,
        }
    )

tokenized_docs = [doc["text"].lower().split() for doc in documents if doc["text"].strip()]

# Avoid constructing BM25 with an empty corpus.
bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None


def bm25_search(
    query: str,
    top_k: int = 5
):

    if bm25 is None or not documents:
        return []

    query_tokens = (
        query.lower()
        .split()
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for idx, score in ranked[:top_k]:

        results.append({
            "source":
                documents[idx]["source"],
            "text":
                documents[idx]["text"],
            "score":
                float(score)
        })

    return results