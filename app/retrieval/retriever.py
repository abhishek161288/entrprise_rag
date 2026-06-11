from qdrant_client import QdrantClient

from app.config import (
    COLLECTION_NAME,
    get_embedding_model
)

client = QdrantClient(
    host="localhost",
    port=6333
)

model = get_embedding_model()

def search_documents(question, limit=5):
    query_embedding = model.encode(
        question
    ).tolist()

    results = client.query_points(
        collection_name=
        COLLECTION_NAME,

        query=query_embedding,

        limit=limit
    )

    documents = []

    for point in results.points:

        documents.append(
            {
                "score":
                    point.score,

                "source":
                    point.payload[
                        "source"
                    ],

                "text":
                    point.payload[
                        "text"
                    ]
            }
        )

    return documents

if __name__ == "__main__":

    question = input(
        "Question: "
    )

    results = search_documents(
        question
    )

    for doc in results:

        print("\n")
        print(
            "=" * 50
        )

        print(
            f"Score: "
            f"{doc['score']}"
        )

        print(
            f"Source: "
            f"{doc['source']}"
        )

        print(
            doc['text']
        )