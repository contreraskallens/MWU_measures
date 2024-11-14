"""
This module takes a preprocessed corpus and builds the frequency 
data structures needed to extract the MWU variables.
"""

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk import FreqDist, bigrams
from . import preprocessing_corpus

BIGRAM_PER_CORPUS = None
CORPUS_PROPORTIONS = None
UNIGRAM_FREQUENCIES_PC = None
UNIGRAM_TOTAL = None
BIGRAM_FW = None
BIGRAM_BW = None

def get_corpus_props(unigram_freqs_pc):
### STILL WORKS
    """
    Gets the proportion of the total unigrams that each corpus has. 
    Necessary for obtaining dispersion measure.
    """
    corpus_sizes = {corpus: dist.total() for corpus, dist in unigram_freqs_pc.items()}
    corpus_total = np.sum(list(corpus_sizes.values()))
    corpus_props = [(corpus, size / corpus_total) for corpus, size in corpus_sizes.items()]
    corpus_props = pd.DataFrame(corpus_props, columns=['corpus', 'corpus_prop'])
    return corpus_props


def process_corpus(
    corpus='bnc',
    corpus_dir=None,
    verbose=False,
    test_corpus=False,
    chunk_size = 10000
    ):
## TODO RETOOL FOR DOING PROCESSING IN THE OTHER SIDE
## MAYBE PREPROCESSING COULD JUST BE THE FUNCTIONS TO GO FROM LINE -> (Corpus, Clean_Line)?
    """
    Takes preprocessed corpus and outputs the data structures necessary to compute MWU measures.
    The data obtained are frequencies for unigrams and bigrams, 
        proportion of unigrams for each corpus,
    and bigram dictionaries of the form {Corpus: {Unigram1: nltk.FreqDist}}.
    :param corpus: The name of the corpus. For now, must be hardcoded. This
        determines the preprocessing routine to perform.
    :param corpus_dir: The directory of the corpus file.
    :param verbose: Whether to print progress reports.
    :param test_corpus: If True, the script is run on the synthetic corpus 
        provided by S. Gries in the original paper. Useful for testing 
        the measures calculated.
    :param chunk_size: In bytes, the size of each chunk from the corpus 
        file to be processed at once.
    :returns: Does not return anything. Instead, it sets global variables
        UNIGRAM_FREQUENCIES_PC, BIGRAM_PER_CORPUS, UNIGRAM_TOTAL,
        BIGRAM_FW, BIGRAM_BW, CORPUS_PROPORTIONS
    """
    # TODO: consider making it return something and not use global scope variables.

    global UNIGRAM_FREQUENCIES_PC
    global UNIGRAM_TOTAL
    global TRIGRAM_FW
    global TRIGRAM_BW
    global TRIGRAM_MERGED_BW
    global CORPUS_PROPORTIONS

    # TODO: should make brown the default corpus because it's included in nltk

    if verbose:
        print('Getting everything ready for score extraction')
    if corpus == 'bnc' and corpus_dir:
        UNIGRAM_FREQUENCIES_PC, TRIGRAM_FW, TRIGRAM_BW, TRIGRAM_MERGED_BW = preprocessing_corpus.preprocess_bnc(
            corpus_dir,
            chunk_size=chunk_size,
            verbose=verbose
            )
    if test_corpus:
        # TODO fix test corpus with new procedure
        corpus_a = 'a d c b e b f g h c b i j k a y z b n o a c c b p q r q a x r z n a'.split()
        corpus_b = 'y i b c p x e j d g n k q r b x x c b d y z f o p q b d j e z b d'.split()
        corpus_c = 'g g i o r j j b c d g j k r e j g f h k h f d h k o a c b r d g k b'.split()

        UNIGRAM_FREQUENCIES_PC = {
            'A': FreqDist(corpus_a),
            'B': FreqDist(corpus_b),
            'C': FreqDist(corpus_c)
            }
        # BIGRAM_PER_CORPUS = {
        #     'A': FreqDist(bigrams(corpus_a)),
        #     'B': FreqDist(bigrams(corpus_b)),
        #     'C': FreqDist(bigrams(corpus_c))
        #     }
    #else: brown corpus

    UNIGRAM_TOTAL = sum(UNIGRAM_FREQUENCIES_PC.values(), Counter())
    CORPUS_PROPORTIONS = get_corpus_props(UNIGRAM_FREQUENCIES_PC)
