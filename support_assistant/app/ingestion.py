from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("doc_*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        documents.append(
            {
                "id": file_path.stem,
                "text": text,
            }
        )

    return documents


def build_collection():
    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        embeddings=embeddings,
    )

    print(f"Loaded {len(documents)} documents.")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Stored at: {CHROMA_DIR}")


if __name__ == "__main__":
    build_collection()