"""
RAG Module for ZAC 2026 LifeMate:
- exemplar_retriever.py: BM25 Sparse Retrieval on train dataset (Person 1)
- history_resolver.py: Context & History Entity Resolver (Person 2)
"""

from .exemplar_retriever import BM25ExemplarRetriever

__all__ = ["BM25ExemplarRetriever"]
