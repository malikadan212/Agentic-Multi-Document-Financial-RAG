"""
Evaluation Module for Financial RAG System
Implements metrics: Recall@K, Exact Match, F1, ROUGE, BERTScore, Citation Accuracy
"""

from .evaluator import (
    TestCase,
    EvaluationResult,
    RetrievalEvaluator,
    GenerationEvaluator,
    CitationEvaluator,
    ComprehensiveEvaluator
)

__all__ = [
    'TestCase',
    'EvaluationResult',
    'RetrievalEvaluator',
    'GenerationEvaluator',
    'CitationEvaluator',
    'ComprehensiveEvaluator'
]
