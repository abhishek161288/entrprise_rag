from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "enterprise_docs"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5" #"all-MiniLM-L6-v2"

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


def get_embedding_model():
    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )