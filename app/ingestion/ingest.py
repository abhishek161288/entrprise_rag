import os

from pathlib import Path

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from app.config import (
    COLLECTION_NAME,
    get_embedding_model
)

DATA_FOLDER = "knowledge_base"

# Load Documents
def load_documents():

    documents = []

    for file in Path(DATA_FOLDER).glob("*.txt"):

        with open(file, "r") as f:

            content = f.read()

            documents.append({
                "source": file.name,
                "content": content
            })

    return documents

# Chunk Documents
def chunk_text(text,
               chunk_size=500):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start = end

    return chunks


#Connect Qdrant
client = QdrantClient(
    host="localhost",
    port=6333
)

# Create CollectioN
def create_collection():

    collections = client.get_collections()

    names = [
        c.name
        for c in collections.collections
    ]

    #if COLLECTION_NAME not in names:

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

# model = get_embedding_model()

# embedding = model.encode(
#     chunk
# ).tolist()

# points.append(

#     PointStruct(

#         id=counter,

#         vector=embedding,

#         payload={

#             "text": chunk,

#             "source": doc["source"]

#         }
#     )
# )

# client.upsert(

#     collection_name=COLLECTION_NAME,

#     points=points
# )

def main():

    create_collection()

    docs = load_documents()

    model = get_embedding_model()

    points = []

    counter = 1

    for doc in docs:

        chunks = chunk_text(
            doc["content"]
        )

        for chunk in chunks:

            embedding = model.encode(
                chunk
            ).tolist()

            points.append(

                PointStruct(
                    id=counter,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "source":
                        doc["source"]
                    }
                )
            )

            counter += 1

    client.upsert(
        collection_name=
        COLLECTION_NAME,
        points=points
    )

    print(
        f"Indexed {len(points)} chunks"
    )


if __name__ == "__main__":
    main()