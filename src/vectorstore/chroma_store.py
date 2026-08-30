# src/vectorstore/chroma_store.py
import chromadb
from chromadb.utils import embedding_functions

from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    TOP_K_RESULTS,
    OLLAMA_HOST,
    OLLAMA_EMBED_MODEL,
)

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Chroma's DefaultEmbeddingFunction downloads its ONNX model from S3 on first
# use, which breaks the offline requirement. Route embeddings through the
# local Ollama server instead: run `ollama pull nomic-embed-text` once during
# setup and nothing needs the network again after that.
_embedding_fn = embedding_functions.OllamaEmbeddingFunction(
    url=OLLAMA_HOST,
    model_name=OLLAMA_EMBED_MODEL,
)

_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=_embedding_fn,
)


def add_document(doc_id: str, text: str, metadata: dict) -> None:
    """
    Store one page. `text` is the Florence-2 caption plus OCR text, combined,
    since that is what we search against. `metadata` should include at least
    image_path, page_number, and source_file so we can show and cite it later.
    """
    _collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata],
    )


def add_documents_batch(doc_ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    """Same as add_document, batched. Use this for ingesting a whole PDF at once."""
    _collection.add(ids=doc_ids, documents=texts, metadatas=metadatas)


def query(question: str, k: int = None) -> list[dict]:
    """
    Return the top-k closest pages to the question.
    Each result: {"text": str, "metadata": dict, "distance": float}
    """
    k = k or TOP_K_RESULTS
    results = _collection.query(query_texts=[question], n_results=k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


def collection_count() -> int:
    """How many pages are currently indexed. Useful for a sanity check in the UI."""
    return _collection.count()


def reset_collection() -> None:
    """Wipe everything. Useful during development, dangerous in production."""
    _client.delete_collection(CHROMA_COLLECTION_NAME)
    global _collection
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )