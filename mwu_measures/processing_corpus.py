"""
This module takes a preprocessed corpus and builds the frequency 
data structures needed to extract the MWU variables.
"""

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk import flatten
from . import preprocessing_corpus

BIGRAM_PER_CORPUS = None
CORPUS_PROPORTIONS = None
UNIGRAM_FREQUENCIES_PC = None
UNIGRAM_TOTAL = None
BIGRAM_FW = None
BIGRAM_BW = None

# def get_corpus_props(unigram_freqs_pc):
# ### STILL WORKS
#     """
#     Gets the proportion of the total unigrams that each corpus has. 
#     Necessary for obtaining dispersion measure.
#     """
#     corpus_sizes = {corpus: dist.total() for corpus, dist in unigram_freqs_pc.items()}
#     corpus_total = np.sum(list(corpus_sizes.values()))
#     corpus_props = [(corpus, size / corpus_total) for corpus, size in corpus_sizes.items()]
#     corpus_props = pd.DataFrame(corpus_props, columns=['corpus', 'corpus_prop'])
#     return corpus_props



# class Corpus():
#     def __init__(self, corpus_name):
#         self.corpus_name = corpus_name
#         self.trigram_freqs = defaultdict(Counter)
#         self.unigram_freqs = defaultdict(Counter)
#         self.n_trigrams = 0
#         self.n_unigrams = 0
#         self.unigram_total = Counter()
#     def add_chunk(self, line_chunk):
#         for corpus, unigrams in line_chunk[0].items():
#             self.unigram_freqs[corpus].update(unigrams)
#             self.n_unigrams += len(unigrams)
#             self.unigram_total.update(unigrams)
#         for corpus, trigrams in line_chunk[1].items():
#             self.trigram_freqs[corpus].update(trigrams)
#             self.n_trigrams += len(trigrams)
#     def query_ngram(self, ngram):
#         ngram = ngram.split()
#         if len(ngram) == 2:
#             counts = defaultdict(Counter)
#             for corpus, corpus_dict in self.trigram_freqs.items():
#                 # TODO: ugly :/
#                 for trigram, frequency in corpus_dict.items():
#                     if trigram[0] == ngram[0]:
#                         counts[corpus].update({trigram[1]: frequency})
#         if len(ngram) == 3:
#             counts = {corpus: Counter({trigram[2]: frequency for trigram, frequency in corpus_dict.items() if (trigram[0], trigram[1]) == (ngram[0], ngram[1])}) for corpus, corpus_dict in self.trigram_freqs.items()}
#         return counts
#     def query_inverse_ngram(self, ngram):
#         ngram = ngram.split()
#         if len(ngram) == 2:
#             counts = defaultdict(Counter)
#             for corpus, corpus_dict in self.trigram_freqs.items():
#                 # TODO: ugly :/
#                 for trigram, frequency in corpus_dict.items():
#                     if trigram[1] == ngram[1]:
#                         counts[corpus].update({trigram[0]: frequency})
#         if len(ngram) == 3:
#             counts = {corpus: Counter({(trigram[0], trigram[1]): frequency for trigram, frequency in corpus_dict.items() if trigram[2] == ngram[2]}) for corpus, corpus_dict in self.trigram_freqs.items()}
#         return counts

#     def set_corpus_props(self):
#         """
#         Gets the proportion of the total unigrams that each corpus has. 
#         Necessary for obtaining dispersion measure.
#         """
#         print("Computing corpus proportions...")
#         corpus_sizes = {corpus: dist.total() for corpus, dist in self.unigram_freqs.items()}
#         corpus_total = np.sum(list(corpus_sizes.values()))
#         corpus_props = [(corpus, size / corpus_total) for corpus, size in corpus_sizes.items()]
#         corpus_props = pd.DataFrame(corpus_props, columns=['corpus', 'corpus_prop'])
#         self.corpus_proportions = corpus_props
#     def set_totals(self):
#         print("Computing unigram and trigram totals...")
#         self.unigram_total = sum(self.unigram_freqs.values(), Counter())
#         self.n_trigrams = sum(self.trigram_freqs.values(), Counter()).total()
#     # Clean? 1 appearance per corpus, caput? This would limit the lookup time by a lot
    
#     def reduce_frequency(self, ngrams):
#         occurring_unigrams = flatten([ngram.split() for ngram in ngrams])
#         reduced_trigrams = {corpus: {trigram: frequency for trigram, frequency in corpus_dict if any(unigram in occurring_unigrams for unigram in trigram)} for corpus, corpus_dict in self.trigram_freqs.items()}
#         return reduced_trigrams



def process_corpus(
        corpus_name='bnc',
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
    this_corpus = Corpus(corpus_name)
    if corpus_name == 'bnc' and corpus_dir:
        with open(corpus_dir, 'r', encoding="utf-8") as corpus_file:
            i = 0
            while True:
                raw_lines = corpus_file.readlines(chunk_size)
                if not raw_lines:
                    break
                ngram_dicts = preprocessing_corpus.preprocess_bnc(raw_lines)
                this_corpus.add_chunk(ngram_dicts)
                if verbose:
                    i += len(raw_lines)
                    print(f'{i} lines processed')
                
    this_corpus.set_corpus_props()
    # this_corpus.set_totals()

    return this_corpus
    # global UNIGRAM_FREQUENCIES_PC
    # global UNIGRAM_TOTAL
    # global TRIGRAM_FW
    # global TRIGRAM_BW
    # global TRIGRAM_MERGED_BW
    # global CORPUS_PROPORTIONS
    # global N_TRIGRAMS
    # # TODO: should make brown the default corpus because it's included in nltk

    # if verbose:
    #     print('Getting everything ready for score extraction')
    # if corpus == 'bnc' and corpus_dir:
    #     UNIGRAM_FREQUENCIES_PC, N_TRIGRAMS, TRIGRAM_FW, TRIGRAM_BW, TRIGRAM_MERGED_BW = preprocessing_corpus.preprocess_bnc(
    #         corpus_dir,
    #         chunk_size=chunk_size,
    #         verbose=verbose
    #         )
    # if test_corpus:
    #     UNIGRAM_FREQUENCIES_PC, N_TRIGRAMS, TRIGRAM_FW, TRIGRAM_BW, TRIGRAM_MERGED_BW = preprocessing_corpus.preprocess_test()
    # #else: brown corpus
    
    # UNIGRAM_TOTAL = sum(UNIGRAM_FREQUENCIES_PC.values(), Counter())
    # CORPUS_PROPORTIONS = get_corpus_props(UNIGRAM_FREQUENCIES_PC)
