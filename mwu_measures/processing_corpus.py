"""
This module takes a preprocessed corpus and builds the frequency 
data structures needed to extract the MWU variables.
"""

from collections import defaultdict, Counter
from itertools import groupby
import numpy as np
import pandas as pd
from nltk import flatten
from . import preprocessing_corpus
import duckdb
import polars as pl
from line_profiler import LineProfiler
lp = LineProfiler()


class Corpus():
    def __init__(self, corpus_name):
        self.corpus_conn = duckdb.connect(":memory:")
        self.corpus_conn.execute(
            """
            CREATE TABLE trigram_db_unagg (
                corpus TEXT,
                ug_1 TEXT,
                ug_2 TEXT,
                ug_3 TEXT,
                freq INTEGER
                )
        """)
        self.corpus_conn.execute(
            """
            CREATE TABLE unigram_db_unagg (
                corpus TEXT,
                ug TEXT,
                freq INTEGER
                )
        """)

        self.corpus_conn.execute(
            """
            CREATE TABLE trigram_db_unagg_test (
                corpus TEXT,
                ug_1 TEXT,
                ug_2 TEXT,
                ug_3 TEXT,
                freq INTEGER
                )
        """)
        self.corpus_conn.execute(
            """
            CREATE TABLE unigram_db_unagg_test (
                corpus TEXT,
                ug TEXT,
                freq INTEGER
                )
        """)

    def add_chunk(self, ngram_lists):
        chunk_unigrams, chunk_trigrams = ngram_lists
        chunk_unigrams = pd.DataFrame(chunk_unigrams, columns=['corpus', 'ug', 'freq'])
        chunk_trigrams = pd.DataFrame(chunk_trigrams, columns=['corpus', 'ug_1', 'ug_2', 'ug_3', 'freq'])
        self.corpus_conn.execute("INSERT INTO unigram_db_unagg SELECT * FROM chunk_unigrams")
        self.corpus_conn.execute("INSERT INTO trigram_db_unagg SELECT * FROM chunk_trigrams")
        self.corpus_conn.execute("VACUUM ANALYZE")


    def consolidate_corpus(self, threshold=1):
        self.corpus_conn.execute(
            """
            CREATE TABLE drop_table_trigrams AS(
                SELECT 
                    CONCAT_WS(' ', ug_1, ug_2, ug_3) AS ngram,
                    SUM(freq) as freq
                FROM trigram_db_unagg
                GROUP BY(ngram)
            )
        """)
        self.corpus_conn.execute(
            f"""
            CREATE OR REPLACE TABLE drop_table_trigrams AS(
            SELECT 
                *, 
                hash(ngram) AS ngram_hash
            FROM drop_table_trigrams
            WHERE freq <= {threshold}
        )
        """)

        self.corpus_conn.execute(
            """
            CREATE TABLE drop_table_bigrams AS (
                SELECT 
                    CONCAT_WS(' ', ug_1, ug_2) AS ngram,
                    SUM(freq) as freq
                FROM trigram_db_unagg
                GROUP BY(ngram)
            )
        """)
        self.corpus_conn.execute(
            f"""
            CREATE OR REPLACE TABLE drop_table_bigrams AS(
            SELECT 
                *, 
                hash(ngram) AS ngram_hash
            FROM drop_table_bigrams
            WHERE freq <= {threshold}
        )
        """)

        self.corpus_conn.execute(
            """
            CREATE TABLE trigram_db AS
            SELECT 
                corpus, 
                ug_1,
                ug_2,
                ug_3,
                CONCAT_WS(' ', ug_1, ug_2) AS big_1,
                CONCAT_WS(' ', ug_1, ug_2, ug_3) as ngram,
                SUM(freq) AS freq
            FROM trigram_db_unagg
            GROUP BY 
                corpus,
                ug_1,
                ug_2,
                ug_3,
                big_1
            ORDER BY
                corpus,
                ug_1,
                ug_2,
                ug_3
        """)
        self.corpus_conn.execute(
            """
            CREATE TABLE unigram_db AS
            SELECT 
                corpus,
                ug,
                SUM(freq) AS freq
            FROM unigram_db_unagg
            GROUP BY
                corpus,
                ug
            ORDER BY 
                corpus,
                ug
        """)
        self.corpus_conn.execute(
            """
            ALTER TABLE unigram_db ADD
                hash_index UINT64
            """)
        self.corpus_conn.execute("""
            UPDATE unigram_db
            SET 
                hash_index = hash(ug)
        """)
        self.corpus_conn.execute("DROP TABLE unigram_db_unagg")
        self.corpus_conn.execute("DROP TABLE trigram_db_unagg")
        # total unigrams
        ug_freqs = self.corpus_conn.execute(
            """
            SELECT
                ug,
                SUM(freq) AS freq
            FROM unigram_db 
            GROUP BY ug
        """)
        ug_freqs = ug_freqs.fetch_df()
        self.total_unigrams = Counter(dict(zip(ug_freqs.ug, ug_freqs.freq)))
        # corpus proportions
        self.corpus_conn.execute(
            """
            CREATE TABLE corpus_proportions AS
                SELECT 
                    corpus,
                    SUM(freq) / (SELECT SUM(freq) FROM unigram_db) AS corpus_prop
                FROM unigram_db 
                GROUP BY corpus
        """)
        self.corpus_proportions = self.corpus_conn.execute("SELECT * FROM corpus_proportions")
        self.corpus_proportions = self.corpus_proportions.fetch_df()
        self.n_trigrams = self.corpus_conn.execute("SELECT SUM(freq) FROM trigram_db").fetchone()[0]
        
        self.corpus_conn.execute("VACUUM ANALYZE")
    def create_totals(self):
        trigram_maxes = self.corpus_conn.execute(
            """
            WITH trigram_totals AS (
                SELECT 
                    ug_1,
                    ug_2,
                    ug_3,
                    SUM(freq) as freq
                FROM trigram_db
                GROUP BY
                    ug_1,
                    ug_2,
                    ug_3
            ), token_frequency AS (
                SELECT 
                    max(freq) AS max_token_trigram
                FROM trigram_totals
            ), type_1 AS (
                SELECT max(typef_1) as max_type1_trigram
                FROM (
                    SELECT 
                        ug_3,
                        count( * ) AS typef_1
                    FROM trigram_totals
                    GROUP BY ug_3
                )
            ), type_2 AS (
                SELECT 
                    max(typef_2) AS max_type2_trigram
                FROM (
                    SELECT
                        ug_1,
                        ug_2,
                        count( * ) AS typef_2
                    FROM trigram_totals
                    GROUP BY
                        ug_1,
                        ug_2
                )
            )
            SELECT
                token_frequency.max_token_trigram,
                type_1.max_type1_trigram,
                type_2.max_type2_trigram
            FROM 
                token_frequency,
                type_1,
                type_2
        """)
        trigram_maxes = trigram_maxes.fetch_df().iloc[0]

        bigram_maxes = self.corpus_conn.execute(
            """
            WITH bigram_totals AS (
                SELECT
                    ug_1,
                    ug_2,
                    SUM(freq) as freq
                FROM trigram_db
                GROUP BY 
                    ug_1,
                    ug_2
            ), token_frequency AS (
                SELECT 
                    max(freq) AS max_token_bigram
                FROM bigram_totals
            ), type_1 AS (
                SELECT 
                    max(typef_1) AS max_type1_bigram
                FROM (
                    SELECT
                        ug_2,
                        count( * ) AS typef_1
                    FROM bigram_totals
                    GROUP BY ug_2
                )
            ), type_2 AS (
                SELECT max(typef_2) AS max_type2_bigram
                FROM (
                    SELECT ug_1,
                    count( * ) AS typef_2
                FROM bigram_totals
                GROUP BY ug_1
                )
            )
            SELECT
                token_frequency.max_token_bigram,
                type_1.max_type1_bigram,
                type_2.max_type2_bigram
            FROM 
                token_frequency,
                type_1,
                type_2
        """)
        bigram_maxes = bigram_maxes.fetch_df().iloc[0]
        self.max_freqs = pd.concat([bigram_maxes, trigram_maxes])
        
    def set_getting_functions(self):
        print("we bring the boom")
        # self.corpus_conn.execute(
        #     """
        #     CREATE OR REPLACE TEMPORARY TABLE bigram_fw_query
        #         (query TEXT,
        #         corpus TEXT,
        #         ug_2 TEXT,
        #         freq INT)
        # """)
        # self.corpus_conn.execute(
        #     """
        #     CREATE OR REPLACE TEMPORARY TABLE trigram_fw_query
        #         (query TEXT,
        #         corpus TEXT,
        #         ug_3 TEXT,
        #         freq INT)
        # """)
        # self.fw_bigram_query = """
        #     INSERT INTO bigram_fw_query
        #     SELECT 
        #         ? AS query,
        #         corpus,
        #         ug_2,
        #         SUM(freq) AS freq 
        #     FROM trigram_db WHERE ug_1 = ?
        #     GROUP BY
        #         corpus,
        #         ug_2
        # """
        # self.fw_trigram_query = """
        #     INSERT INTO trigram_fw_query
        #     SELECT
        #         ? AS query,
        #         corpus,
        #         ug_3,
        #         freq
        #     FROM trigram_db 
        #     WHERE ug_1 = ? AND ug_2 = ?
        # """
        # self.corpus_conn.execute("""
        #     CREATE OR REPLACE TEMPORARY TABLE bigram_bw_query 
        #                          (query TEXT,
        #                          corpus TEXT,
        #                          ug_2 TEXT,
        #                          freq INT)
        #     """)
        # self.corpus_conn.execute("""
        #     CREATE OR REPLACE TEMPORARY TABLE trigram_bw_query
        #                         (query TEXT,
        #                         corpus TEXT,
        #                         ug_1 TEXT,
        #                         ug_2 TEXT,
        #                         freq INT)
        # """)
        # self.bw_bigram_query = """
        #     INSERT INTO bigram_bw_query
        #     SELECT 
        #         ? AS query,
        #         corpus,
        #         ug_1,
        #         SUM(freq) AS freq
        #     FROM trigram_db
        #     WHERE ug_2 = ?
        #     GROUP BY
        #         corpus,
        #         ug_2
        # """
        # self.bw_trigram_query = """
        #     INSERT INTO trigram_bw_query
        #     SELECT
        #         ? AS query,
        #         corpus,
        #         ug_1,
        #         ug_2,
        #         freq from trigram_db
        #     WHERE ug_3 = ?
        # """
        # self.unigram_query = """
        #     SELECT
        #         corpus,
        #         ug,
        #         freq
        #     FROM unigram_db 
        #     WHERE ug = ?
        # """

    def get_fw_distribution(self, ngram):
        ngrams = ngram.split()
        if len(ngrams) == 2:
            queried_ngram = self.corpus_conn.execute(self.fw_bigram_query, [ngrams[0]]).fetchall()
            distribution = {corpus: Counter({v[2]: v[3] for v in vs}) for corpus, vs in groupby(queried_ngram, lambda x: x[0])}
        if len(ngrams) == 3:
            queried_ngram = self.corpus_conn.execute(self.fw_trigram_query, [ngrams[0], ngrams[1]]).fetchall()
            distribution = {corpus: Counter({v[3]: v[4] for v in vs}) for corpus, vs in groupby(queried_ngram, lambda x: x[0])}
        return distribution

    def get_bw_distribution(self, ngram):
        ngrams = ngram.split()
        if len(ngrams) == 2:
            queried_ngram = self.corpus_conn.execute(self.bw_bigram_query, [ngrams[1]]).fetchall()
            distribution = {corpus: Counter({v[1]: v[3] for v in vs}) for corpus, vs in groupby(queried_ngram, lambda x: x[0])}
        if len(ngrams) == 3:
            queried_ngram = self.corpus_conn.execute(self.bw_trigram_query, [ngrams[2]]).fetchall()
            distribution = {corpus: Counter({(v[1], v[2]): v[4] for v in vs}) for corpus, vs in groupby(queried_ngram, lambda x: x[0])}
        return distribution

    def get_unigram(self, unigram):
        unigram_info = self.corpus_conn.execute(f"EXECUTE get_unigram({unigram})").fetch_df()
        return unigram_info

def process_corpus(
        corpus_name='bnc',
        corpus_dir=None,
        verbose=False,
        test_corpus=False,
        chunk_size = 1000000
        ):
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
        this_corpus.set_getting_functions()
        return this_corpus
    if test_corpus:
        this_corpus = Corpus('test')
        ngram_dicts = preprocessing_corpus.preprocess_test()
        this_corpus.add_chunk(ngram_dicts)
        this_corpus.consolidate_corpus()
        this_corpus.set_getting_functions()
        return this_corpus
    
