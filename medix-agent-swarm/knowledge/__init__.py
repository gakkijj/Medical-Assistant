"""Knowledge package with lightweight retrieval and lazy Milvus loading."""
from .hybrid_retriever import Chunk, HybridRetriever, load_chunks


def __getattr__(name):
    if name != "MedicalKnowledgeBase":
        raise AttributeError(name)
    from .milvus_kb import MedicalKnowledgeBase
    return MedicalKnowledgeBase


__all__ = ["Chunk", "HybridRetriever", "load_chunks", "MedicalKnowledgeBase"]
