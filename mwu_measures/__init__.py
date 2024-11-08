"""
Main loading
"""
from .processing_corpus import process_corpus
from .mwu_functions import get_mwu_scores

__all__ = [
    'process_corpus',
    'get_mwu_scores',
]
