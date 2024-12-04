import pandas as pd
from orjson import loads
import numpy as np
from functools import reduce

default_weights = {'token_freq': 1/8, 'dispersion': 1/8, 'type_1': 1/8, 'type_2': 1/8, 'entropy_1': 1/8, 'entropy_2': 1/8, 'fw_assoc': 1/8, 'bw_assoc': 1/8}

class Fetcher():
    def __init__(self, conn):
        self.conn = conn

    def __call__(self, query):
        return(self.conn.execute(query))

    def allocate_query(self, source, ngrams):
        query_df = pd.DataFrame(ngrams, columns = ['query', source])
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE this_query (query TEXT, {source} TEXT, hash_index UINT64)
        """)
        self.conn.execute(f"""
            INSERT INTO this_query SELECT *, hash({source}) as hash_index FROM query_df
        """)

    def create_query(self, ngrams, source, target):
        query_df = pd.DataFrame(ngrams, columns = [source, target])
    
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE this_query 
            ({source} TEXT, {target} TEXT)
        """)
        self.conn.execute(
            f"""
            INSERT INTO this_query 
            SELECT {source}, {target} 
            FROM query_df
        """)
    def allocate_filtered_db(self, source, target):
            self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS (
                SELECT
                    corpus,
                    {source},
                    {target},
                    SUM(freq) AS freq,
                FROM trigram_db
                WHERE 
                    {source} IN (SELECT {source} FROM this_query) OR 
                    {target} IN (SELECT {target} FROM this_query)
                GROUP BY 
                    corpus,
                    {source},
                    {target}
            )
        """)
            self.conn.execute(
                f"""
                CREATE OR REPLACE TABLE filtered_db_total AS (
                    SELECT
                        {source},
                        {target},
                        SUM(freq) AS total_freq,
                    FROM filtered_db
                    GROUP BY 
                        {source},
                        {target}
                )
            """)
    def make_token_freq(self, source, target):
        # Make a table with token frequencies for the queried ngrams

        self(f"""
        CREATE OR REPLACE TABLE token_freq AS
        SELECT 
            {source},
            {target},
            total_freq AS token_freq
        FROM filtered_db_total
        RIGHT JOIN this_query
        USING({source}, {target})
    """)

    def reduce_query(self, source, target):
        # Make a reduced query table that includes only the ngrams that have appear in the corpus.
        # This is a substantial memory and runtime save for later operations.
        # Then, clean the token frequency table.
        self(
        """
        CREATE OR REPLACE TABLE ignore_ngrams AS
        SELECT * 
        FROM token_freq 
        WHERE token_freq IS NOT DISTINCT FROM NULL
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE reduced_query AS
        SELECT * 
        FROM this_query
        ANTI JOIN ignore_ngrams
        USING({source}, {target})
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE token_freq AS
        SELECT *
        FROM token_freq
        SEMI JOIN reduced_query
        USING({source}, {target})
    """)

    def make_type_freq(self, source, target):
        self(
        f"""
        CREATE OR REPLACE TABLE type_1 AS
        SELECT 
            {target},
            COUNT( * ) AS typef_1
        FROM filtered_db_total
        WHERE {target} IN (SELECT {target} FROM reduced_query)
        GROUP BY
            {target}
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE type_2 AS
        SELECT 
            {source},
            COUNT( * ) AS typef_2
        FROM filtered_db_total
        WHERE {source} IN (SELECT {source} FROM reduced_query)
        GROUP BY
            {source}
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE type_freq AS
        WITH freq_1_reduced AS (
            SELECT *
            FROM reduced_query
            LEFT JOIN type_1
            USING ({target})
        )
        SELECT *
        FROM freq_1_reduced
        LEFT JOIN type_2
        USING ({source})
    """)

    def make_dispersion(self, source, target):
        self(
        f"""
        CREATE OR REPLACE TABLE corpus_ngram_combs AS 
        SELECT corpus, {source}, {target}
        FROM (SELECT DISTINCT {source}, {target} FROM filtered_db)
        SEMI JOIN reduced_query
        USING({source}, {target})
        CROSS JOIN (SELECT DISTINCT corpus FROM filtered_db)
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE disp_table AS
        SELECT corpus, {source}, {target}, COALESCE(freq, 0) as freq
        FROM corpus_ngram_combs
        LEFT JOIN filtered_db
        USING(corpus, {source}, {target})
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE disp_table AS
        SELECT *
        FROM disp_table 
        LEFT JOIN corpus_proportions
        USING(corpus)
        WHERE {source} IN (SELECT {source} FROM this_query) AND {target} IN (SELECT {target} FROM this_query)
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE disp_table AS
        SELECT 
            corpus,
            disp_table.{source},
            disp_table.{target},
            freq / total_freq AS proportion,
            corpus_prop
        FROM disp_table
        LEFT JOIN filtered_db_total
        USING({source}, {target})
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE dispersion AS
        WITH kld_table AS (
            SELECT 
                corpus,
                {source},
                {target},
                CASE
                    WHEN proportion > 0 THEN proportion * log2(proportion / corpus_prop)
                    ELSE 0
                END AS KLD
            FROM disp_table
        )
        SELECT 
            {source},
            {target},
            1 - pow(EXP(1), -SUM(KLD)) as dispersion
        FROM kld_table
        GROUP BY ({source}, {target})
    """)

    def make_associations(self, source, target):
        self(
        """
        CREATE OR REPLACE TABLE unigram_totals AS
        SELECT 
            ug,
            SUM(freq) AS freq,
            SUM(SUM(freq)) OVER () AS total_ug,
        FROM unigram_db
        GROUP BY (ug)
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE rel_freqs AS
        SELECT 
            {source},
            {target},
            token_freq,
            unigram_totals.freq AS source_freq,
            unigram_totals.total_ug 
        FROM token_freq
        LEFT JOIN unigram_totals
        ON token_freq.{source} = unigram_totals.ug
        """)
        self(
        f"""
        CREATE OR REPLACE TABLE rel_freqs AS
        SELECT 
            {source},
            {target},
            token_freq,
            source_freq,
            unigram_totals.freq AS target_freq,
            rel_freqs.total_ug
        FROM rel_freqs
        LEFT JOIN unigram_totals
        ON rel_freqs.{target} = unigram_totals.ug
        """)

        self(
        f"""
        CREATE OR REPLACE TABLE associations AS
        WITH probs_db AS (
            SELECT
                {source},
                {target}, 
                token_freq / source_freq AS prob_2_1,
                token_freq / target_freq AS prob_1_2,
                source_freq / total_ug AS prob_1,
                target_freq / total_ug AS prob_2
            FROM rel_freqs
        ), all_probs AS (
            SELECT
                {source},
                {target},
                prob_2_1,
                prob_1_2,
                prob_1,
                prob_2,
                1 - prob_2_1 AS prob_no_2_1,
                1 - prob_1_2 AS prob_no_1_2,
                1 - prob_1 AS prob_no_1,
                1 - prob_2 AS prob_no_2
            FROM probs_db
        ), forward_kld AS (
            SELECT
                {source},
                {target},
                prob_2_1 * log2(prob_2_1 / prob_2) AS kld_1,
                CASE
                    WHEN prob_no_2_1 = 0 THEN 0
                    ELSE (1 - prob_2_1) * log2((1 - prob_2_1) / (1 - prob_2))
                END AS kld_2
            FROM all_probs
        ), forward_assoc AS (
            SELECT
                {source},
                {target},
                1 - pow(EXP(1), -(kld_1 + kld_2)) AS fw_assoc
            FROM forward_kld
        ), backward_kld AS (
            SELECT
                {source},
                {target},
                prob_1_2 * log2(prob_1_2 / prob_1) AS kld_1,
                CASE
                    WHEN prob_no_1_2 = 0 THEN 0
                    ELSE (1 - prob_1_2) * log2((1 - prob_1_2) / (1 - prob_1))
                END AS kld_2
            FROM all_probs
        ), backward_assoc AS (
            SELECT
                {source},
                {target},
                1 - pow(EXP(1), -(kld_1 + kld_2)) AS bw_assoc
            FROM backward_kld
        )
        SELECT *
        FROM forward_assoc
        LEFT JOIN backward_assoc
        USING({source}, {target})
    """)

    def make_entropy_diff(self, source, target, slot):
        # There has to be a better way of doing all this CTEing for entropies. Functions?
        # The second half of the procedure is the same as the first but with a 
        # different input (0 if query_target == target)
        # Real entropy
        self(
        f"""
        CREATE OR REPLACE TABLE entropy_real_{slot} AS
        WITH all_freqs AS (
            SELECT
                *
            FROM filtered_db_total
            WHERE {source} IN (SELECT {source} FROM reduced_query)
        ), source_freqs AS (
            SELECT *
            FROM reduced_query
            LEFT JOIN all_freqs
            USING({source})
        ), freqs AS (
            SELECT
                {source},
                {target} AS target,
                {target}_1 AS {target},
                total_freq
            FROM source_freqs
        ), totals AS (
            SELECT
                {source},
                SUM(total_freq) AS total
            FROM freqs
            GROUP BY {source}
        ), probs AS (
            SELECT 
                {source},
                target,
                {target},
                total_freq / total AS prob
            FROM freqs
            LEFT JOIN totals
            USING({source})
        ), info AS (
            SELECT
                {source},
                target,
                CASE
                    WHEN prob > 0 THEN prob * log2(prob)
                    ELSE 0
                END AS info
            FROM probs
        ), entropy AS (
            SELECT 
                {source},
                target AS {target},
                SUM(CASE WHEN info < 0 THEN -info ELSE 0 END) AS entropy,
                log2(COUNT (*)) AS normalizer
            FROM info
            GROUP BY {source}, target
        )
        SELECT
            {source},
            {target},
            CASE
                WHEN normalizer > 0 THEN entropy / normalizer
                ELSE 0
            END as entropy_real
        FROM entropy
        ORDER BY entropy_real DESC
    """)

    # Counterfactual entropy

        self(
        f"""
        CREATE OR REPLACE TABLE entropy_cf_{slot} AS
        WITH all_freqs AS (
            SELECT
                *
            FROM filtered_db_total
            WHERE {source} IN (SELECT {source} FROM reduced_query)
        ), source_freqs AS (
            SELECT *
            FROM reduced_query
            LEFT JOIN all_freqs
            USING({source})
        )
        , freqs AS (
            SELECT
                {source},
                {target} AS target,
                {target}_1 AS {target},
                CASE
                    WHEN {target} = {target}_1 THEN 0
                    ELSE total_freq
                END AS total_freq
            FROM source_freqs
        ), totals AS (
            SELECT
                {source},
                SUM(total_freq) AS total
            FROM freqs
            GROUP BY {source}
        ), probs AS (
            SELECT 
                {source},
                target,
                {target},
                total_freq / total AS prob
            FROM freqs
            LEFT JOIN totals
            USING({source})
        ), info AS (
            SELECT
                {source},
                target,
                CASE
                    WHEN prob > 0 THEN prob * log2(prob)
                    ELSE 0
                END AS info
            FROM probs
            WHERE target != {target}
        ), entropy AS (
            SELECT 
                {source},
                target AS {target},
                SUM(CASE WHEN info < 0 THEN -info ELSE 0 END) AS entropy,
                log2(COUNT (*)) AS normalizer
            FROM info
            GROUP BY {source}, target
        )
        SELECT
            {source},
            {target},
            CASE
                WHEN normalizer > 0 THEN entropy / normalizer
                ELSE 0
            END as entropy_cf
        FROM entropy
    """)
        self(
        f"""
        CREATE OR REPLACE TABLE entropy_{slot} AS
        SELECT 
            {source},
            {target},
            entropy_cf - entropy_real AS entropy_{slot}
        FROM entropy_real_{slot}
        INNER JOIN entropy_cf_{slot}
        USING({source}, {target})
    """)

    def make_entropy_diffs(self, source, target):
        self.make_entropy_diff(source, target, '2')
        self.make_entropy_diff(target, source, '1')
        #now it's good
        self(
        f"""
        CREATE OR REPLACE TABLE entropy_diffs AS
        SELECT * FROM entropy_1
        INNER JOIN entropy_2
        USING({source}, {target})
        """)

    def join_measures(self, source, target, length):
        self(
        f"""
        CREATE OR REPLACE TABLE raw_measures AS
        SELECT 
            *,
            {length} AS ngram_length
        FROM reduced_query
        INNER JOIN token_freq USING ({source}, {target})
        INNER JOIN dispersion USING ({source}, {target})
        INNER JOIN type_freq USING ({source}, {target})
        INNER JOIN entropy_diffs USING ({source}, {target})
        INNER JOIN associations USING ({source}, {target})
       """)

    def normalize_measures(self, source, target, entropy_limits=[-0.1, 0.1]):
        self(
            f"""
            CREATE OR REPLACE TABLE normalized_measures AS
            SELECT
                {source},
                {target},
                (log(token_freq) - log(1)) / (log(ngram_totals.max_token) - log(1)) AS token_freq,
                1 - dispersion AS dispersion,
                (log(typef_1) - log(1)) / (log(ngram_totals.max_type1) - log(1)) AS type_1,
                (log(typef_2) - log(1)) / (log(ngram_totals.max_type2) - log(1)) AS type_2,
            CASE
                WHEN entropy_1 < {entropy_limits[0]} THEN 0
                WHEN entropy_1 > {entropy_limits[1]} THEN 1
                ELSE (entropy_1 - {entropy_limits[0]}) / ({entropy_limits[1]} - {entropy_limits[0]})
            END AS entropy_1,
            CASE
                WHEN entropy_2 < {entropy_limits[0]} THEN 0
                WHEN entropy_2 > {entropy_limits[1]} THEN 1
                ELSE (entropy_2 - {entropy_limits[0]}) / ({entropy_limits[1]} - {entropy_limits[0]})
            END AS entropy_2,
            fw_assoc,
            bw_assoc,
            ngram_length
        FROM raw_measures
        LEFT JOIN ngram_totals
        USING (ngram_length)
    """
    )

    def get_mwu_score(self, source, target, weight_dict=default_weights):
        self(
        f"""
        CREATE OR REPLACE TABLE mwu_scores AS
        SELECT
            {source},
            {target},
            {weight_dict['token_freq']} * (token_freq) + 
                {weight_dict['dispersion']} * (dispersion) + 
                {weight_dict['type_1']} * type_1 + 
                {weight_dict['type_2']} * type_2 + 
                {weight_dict['entropy_1']} * entropy_1 + 
                {weight_dict['entropy_2']} * entropy_2 + 
                {weight_dict['fw_assoc']} * fw_assoc + 
                {weight_dict['bw_assoc']} * bw_assoc 
                AS MWU_score,
            ngram_length
        FROM normalized_measures
    """)
        if weight_dict != default_weights:
            self(
            f"""
            CREATE OR REPLACE TABLE default_scores AS
            SELECT
                {source},
                {target},
                {default_weights['token_freq']} * (token_freq) + 
                    {default_weights['dispersion']} * (dispersion) + 
                    {default_weights['type_1']} * type_1 + 
                    {default_weights['type_2']} * type_2 + 
                    {default_weights['entropy_1']} * entropy_1 + 
                    {default_weights['entropy_2']} * entropy_2 + 
                    {default_weights['fw_assoc']} * fw_assoc + 
                    {default_weights['bw_assoc']} * bw_assoc 
                    AS MWU_score
                ngram_length
            FROM normalized_measures
        """)    

    def get_ngram_scores(self, source, target, length, entropy_limits=[-0.1, 0.1], weight_dict=default_weights):
        self.make_token_freq(source, target)
        self.reduce_query(source, target)
        self.make_type_freq(source, target)
        self.make_dispersion(source, target)
        self.make_associations(source, target)
        self.make_entropy_diffs(source, target)
        self.join_measures(source, target, length)
        self.normalize_measures(source, target, entropy_limits)
        self.get_mwu_score(source, target, weight_dict)
        raw_measures = self("SELECT * FROM raw_measures").fetch_df()
        normalized_measures = self("SELECT * FROM normalized_measures").fetch_df()
        mwu_scores = self("SELECT * FROM mwu_scores").fetch_df()
        if weight_dict != default_weights:
            default_scores = self("SELECT * FROm default_scores").fetch_df()
        else:
            default_scores = None
        return {
            'raw': raw_measures,
            'normalized': normalized_measures,
            'scores': mwu_scores,
            'default': default_scores
            }