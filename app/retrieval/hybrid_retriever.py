# app/retrieval/hybrid_retriever.py

from app.retrieval.retriever import (
    search_documents
)

from app.retrieval.bm25_index import (
    bm25_search
)

from app.retrieval.reranker import (
    rerank_documents
)


def hybrid_search(
    question: str,
    vector_limit: int = 10,
    bm25_limit: int = 10,
    final_limit: int = 5
):
    """
    Hybrid Retrieval

    Vector Search
          +
    BM25 Search
          +
    CrossEncoder Re-ranking
    """

    vector_results = search_documents(
        question,
        limit=vector_limit
    )

    keyword_results = bm25_search(
        question,
        top_k=bm25_limit
    )

    merged = {}

    for doc in vector_results:

        key = (
            doc["source"]
            + "_"
            + str(
                hash(
                    doc["text"]
                )
            )
        )

        merged[key] = doc

    for doc in keyword_results:

        key = (
            doc["source"]
            + "_"
            + str(
                hash(
                    doc["text"]
                )
            )
        )

        if key not in merged:
            merged[key] = doc

    candidates = list(
        merged.values()
    )

    ranked_docs = rerank_documents(
        question,
        candidates,
        top_k=final_limit
    )

    return ranked_docs