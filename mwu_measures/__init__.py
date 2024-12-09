"""
Main loading
"""
from .processing_corpus import get_processed_corpus
from .mwu_functions import get_mwu_scores

__all__ = [
    'get_processed_corpus',
    'get_mwu_scores',
]
