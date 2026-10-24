"""RAG (Retrieval-Augmented Generation) subsystem for the counselling chatbot.

Public interface:
    from src.rag import RAGPipeline
    pipeline = RAGPipeline.from_config(config_dict, project_root)
    results  = pipeline.retrieve(query, top_k=5)
"""

from src.rag.rag_pipeline import RAGPipeline  # noqa: F401

__all__ = ["RAGPipeline"]
