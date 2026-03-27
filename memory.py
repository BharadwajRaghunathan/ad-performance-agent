"""
memory.py — Chroma vector DB memory for Ad Performance Agent.

All functions are resilient no-ops if Chroma is unavailable.
"""

import datetime

_chroma_client = None
_collection = None
_embeddings = None
_chroma_available = False

try:
    import chromadb
    from chromadb import PersistentClient
    from langchain_huggingface import HuggingFaceEmbeddings

    _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _chroma_client = PersistentClient(path="./chroma_db")
    _collection = _chroma_client.get_or_create_collection(name="ad_analyses")
    _chroma_available = True
    print("[memory] Chroma initialised — ad_analyses collection ready.")
except Exception as e:
    print(f"[memory] Chroma unavailable — memory features disabled. Reason: {e}")


def save_analysis(campaign_name: str, report_text: str) -> bool:
    """
    Persist a completed ad analysis report to the Chroma vector store.

    Args:
        campaign_name: Human-readable label for the analysis (used as metadata)
        report_text: Full markdown report text to embed and store

    Returns:
        True if saved successfully, False if Chroma is unavailable
    """
    if not _chroma_available or _collection is None or _embeddings is None:
        return False

    try:
        timestamp = datetime.datetime.utcnow().isoformat()
        doc_id = f"analysis_{timestamp}"

        embedding = _embeddings.embed_query(report_text[:1000])

        _collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[report_text],
            metadatas=[{"campaign": campaign_name, "timestamp": timestamp}],
        )
        print(f"[memory] Saved analysis '{campaign_name}' as {doc_id}")
        return True
    except Exception as e:
        print(f"[memory] Failed to save analysis: {e}")
        return False


def retrieve_similar(query: str, k: int = 3) -> list[dict]:
    """
    Retrieve the k most similar past analyses from Chroma.

    Args:
        query: Text to search for similar past campaigns
        k: Number of results to return (default 3)

    Returns:
        List of dicts with keys 'document', 'metadata', 'distance'.
        Returns empty list if Chroma is unavailable.
    """
    if not _chroma_available or _collection is None or _embeddings is None:
        return []

    try:
        count = _collection.count()
        if count == 0:
            return []
        query_embedding = _embeddings.embed_query(query)
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, count),
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"document": doc, "metadata": meta, "distance": dist})

        return output
    except Exception as e:
        print(f"[memory] Failed to retrieve similar analyses: {e}")
        return []
