"""
This module takes a preprocessed corpus and builds the frequency 
data structures needed to extract the MWU variables.
"""

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk import flatten
from . import preprocessing_corpus
import duckdb

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



class Corpus():
    def __init__(self, corpus_name):
        self.corpus_conn = duckdb.connect(":memory:")
        self.corpus_conn.execute("""
        CREATE TABLE trigram_db (
            corpus TEXT,
            ug_1 TEXT,
            ug_2 TEXT,
            ug_3 TEXT,
            freq INTEGER
            )
            """)
        self.corpus_conn.execute("""
        CREATE TABLE unigram_db (
            corpus TEXT,
            ug TEXT,
            freq INTEGER
            )
            """)
    def add_chunk(self, ngram_lists):
        chunk_unigrams, chunk_trigrams = ngram_lists
        chunk_unigrams = pd.DataFrame(chunk_unigrams, columns=['corpus', 'ug', 'freq'])
        chunk_trigrams = pd.DataFrame(chunk_trigrams, columns=['corpus', 'ug_1', 'ug_2', 'ug_3', 'freq'])
        self.corpus_conn.register('chunk_unigrams', chunk_unigrams)
        self.corpus_conn.execute("INSERT INTO unigram_db SELECT * FROM chunk_unigrams")
        self.corpus_conn.register('chunk_trigrams', chunk_trigrams)
        self.corpus_conn.execute("INSERT INTO trigram_db SELECT * FROM chunk_trigrams")

    def consolidate_corpus(self):
        self.corpus_conn.execute("""
        CREATE OR REPLACE TABLE trigram_db AS
        SELECT corpus, ug_1, ug_2, ug_3, SUM(freq) AS freq
        FROM trigram_db
        GROUP BY corpus, ug_1, ug_2, ug_3
        """)
        self.corpus_conn.execute("""
        CREATE OR REPLACE TABLE unigram_db AS
        SELECT corpus, ug, SUM(freq) AS freq
        FROM unigram_db
        GROUP BY corpus, ug
        """)

        # total unigrams
        ug_freqs = self.corpus_conn.execute("SELECT ug, SUM(freq) AS freq FROM unigram_db GROUP BY ug").fetch_df()
        self.total_unigrams = Counter(dict(zip(ug_freqs.ug, ug_freqs.freq)))
        # corpus proportions
        self.corpus_proportions = self.corpus_conn.execute("SELECT corpus, SUM(freq) / (SELECT SUM(freq) FROM unigram_db) AS freq FROM unigram_db GROUP BY corpus").fetch_df()
        # n_trigrams
        self.n_trigrams = self.corpus_conn.execute("SELECT SUM(freq) FROM trigram_db").fetchone()[0]

    def create_totals(self):
        trigram_maxes = self.corpus_conn.execute(
            """
            WITH trigram_totals AS (
                SELECT ug_1, ug_2, ug_3, SUM(freq) as freq
                FROM trigram_db
                GROUP BY ug_1, ug_2, ug_3
            ), token_frequency AS (
                SELECT max(freq) AS max_token_trigram
                FROM trigram_totals
            ), type_1 AS (
                SELECT max(typef_1) as max_type1_trigram
                FROM (
                    SELECT ug_3, count( * ) AS typef_1
                    FROM trigram_totals
                    GROUP BY ug_3
                )
            ), type_2 AS (
                SELECT max(typef_2) AS max_type2_trigram
                FROM (
                    SELECT ug_1, ug_2, count( * ) AS typef_2
                    FROM trigram_totals
                    GROUP BY ug_1, ug_2
                )
            )
        SELECT token_frequency.max_token_trigram, type_1.max_type1_trigram, type_2.max_type2_trigram
        FROM token_frequency, type_1, type_2
        """).fetch_df().iloc[0]

        bigram_maxes = self.corpus_conn.execute(
            """
            WITH bigram_totals AS (
                SELECT ug_1, ug_2, SUM(freq) as freq
                FROM trigram_db
                GROUP BY ug_1, ug_2
            ), token_frequency AS (
                SELECT max(freq) AS max_token_bigram
                FROM bigram_totals
            ), type_1 AS (
                SELECT max(typef_1) AS max_type1_bigram
                FROM (
                    SELECT ug_2, count( * ) AS typef_1
                    FROM bigram_totals
                    GROUP BY ug_2
                )
            ), type_2 AS (
            SELECT max(typef_2) AS max_type2_bigram
            FROM (
                SELECT ug_1, count( * ) AS typef_2
                FROM bigram_totals
                GROUP BY ug_1
            )
        )
        SELECT token_frequency.max_token_bigram, type_1.max_type1_bigram, type_2.max_type2_bigram
        FROM token_frequency, type_1, type_2
        """).fetch_df().iloc[0]
        self.max_freqs = pd.concat([bigram_maxes, trigram_maxes])

    def get_fw_distribution(self, ngram):
        ngrams = ngram.split()
        if len(ngrams) == 2:
            distribution = self.corpus_conn.execute("SELECT corpus, ug_1, ug_2, SUM(freq) AS freq FROM trigram_db WHERE ug_1 = ? GROUP BY corpus, ug_1, ug_2", [ngrams[0]]).fetch_df()
            # TODO: messy, just so I don't have to rewrite calc functions
            distribution = distribution.groupby(by='corpus')[['ug_2', 'freq']].apply(lambda x: Counter(dict(zip(x['ug_2'], x['freq']))))
            distribution = distribution.to_dict()
            # distribution = Counter(dict(zip(distribution.ug_2, distribution.freq)))
        if len(ngrams) == 3:
            distribution = self.corpus_conn.execute("SELECT * FROM trigram_db WHERE ug_1 = ? AND ug_2 = ?", [ngrams[0], ngrams[1]]).fetch_df()
            distribution = distribution.groupby(by='corpus')[['ug_3', 'freq']].apply(lambda x: Counter(dict(zip(x['ug_3'], x['freq']))))
            distribution = distribution.to_dict()
            # distribution = Counter(dict(zip(distribution.ug_3, distribution.freq)))
        return distribution

    def get_bw_distribution(self, ngram):
        ngrams = ngram.split()
        if len(ngrams) == 2:
            distribution = self.corpus_conn.execute("SELECT corpus, ug_1, ug_2, SUM(freq) AS freq FROM trigram_db WHERE ug_2 = ? GROUP BY corpus, ug_1, ug_2", [ngrams[1]]).fetch_df()
            distribution = distribution.groupby(by='corpus')[['ug_1', 'freq']].apply(lambda x: Counter(dict(zip(x['ug_1'], x['freq']))))
            distribution = distribution.to_dict()
        if len(ngrams) == 3:
            distribution = self.corpus_conn.execute("SELECT * FROM trigram_db WHERE ug_3 = ?", [ngrams[2]]).fetch_df()
            distribution = distribution.groupby(by='corpus')[['ug_1', 'ug_2', 'freq']].apply(lambda x: Counter(dict(zip(zip(x['ug_1'], x['ug_2']), x['freq']))))
            distribution = distribution.to_dict()
        return distribution

    def get_unigram(self, unigram):
        unigram_info = self.corpus_conn.execute("SELECT corpus, ug, freq  FROM unigram_db WHERE ug = ?", [unigram]).fetch_df()
        return unigram_info

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
    if corpus_name == 'bnc' and corpus_dir:
        this_corpus = Corpus(corpus_name)
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
                    
        this_corpus.consolidate_corpus()
        return this_corpus
    if test_corpus:
        this_corpus = Corpus('test')
        ngram_dicts = preprocessing_corpus.preprocess_test()
        this_corpus.add_chunk(ngram_dicts)
        this_corpus.consolidate_corpus()
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
