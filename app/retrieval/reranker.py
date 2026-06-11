# app/retrieval/reranker.py

from sentence_transformers import CrossEncoder

# Loaded once during startup
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(
    question: str,
    documents: list,
    top_k: int = 5
):
    """
    Re-rank retrieved documents using CrossEncoder.

    Args:
        question: User query
        documents: Retrieved chunks
        top_k: Number of chunks to keep

    Returns:
        Top ranked chunks
    """

    if not documents:
        return []

    pairs = [
        (question, doc["text"])
        for doc in documents
    ]

    scores = reranker_model.predict(pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return documents[:top_k]