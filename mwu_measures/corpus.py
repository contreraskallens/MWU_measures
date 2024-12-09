import duckdb
import pandas as pd

class Corpus():
    def __init__(self, corpus_name):
        self.conn = duckdb.connect(":memory:")
        self.conn.execute(
            """
            CREATE TABLE trigram_db_unagg (
                corpus TEXT,
                ug_1 TEXT,
                ug_2 TEXT,
                ug_3 TEXT,
                freq INTEGER
                )
        """)
        self.conn.execute(
            """
            CREATE TABLE unigram_db_unagg (
                corpus TEXT,
                ug TEXT,
                freq INTEGER
                )
        """)

    def __call__(self, query):
        return(self.conn.execute(query))

    def add_chunk(self, ngram_lists):
        chunk_unigrams, chunk_trigrams = ngram_lists
        chunk_unigrams = pd.DataFrame(chunk_unigrams, columns=['corpus', 'ug', 'freq'])
        chunk_trigrams = pd.DataFrame(chunk_trigrams, columns=['corpus', 'ug_1', 'ug_2', 'ug_3', 'freq'])
        self.conn.register("chunk_unigrams", chunk_unigrams)
        self.conn.register("chunk_trigrams", chunk_trigrams)
        self("INSERT INTO unigram_db_unagg SELECT * FROM chunk_unigrams")
        self("INSERT INTO trigram_db_unagg SELECT * FROM chunk_trigrams")
        self("VACUUM ANALYZE")

    def consolidate_corpus(self, threshold=1):
        self(
            """
            CREATE TABLE drop_table_trigrams AS(
                SELECT 
                    CONCAT_WS(' ', ug_1, ug_2, ug_3) AS ngram,
                    SUM(freq) as freq
                FROM trigram_db_unagg
                GROUP BY(ngram)
            )
        """)
        self(
            f"""
            CREATE OR REPLACE TABLE drop_table_trigrams AS(
            SELECT 
                *, 
                hash(ngram) AS ngram_hash
            FROM drop_table_trigrams
            WHERE freq <= {threshold}
        )
        """)

        self(
            """
            CREATE TABLE drop_table_bigrams AS (
                SELECT 
                    CONCAT_WS(' ', ug_1, ug_2) AS ngram,
                    SUM(freq) as freq
                FROM trigram_db_unagg
                GROUP BY(ngram)
            )
        """)
        self(
            f"""
            CREATE OR REPLACE TABLE drop_table_bigrams AS(
            SELECT 
                *, 
                hash(ngram) AS ngram_hash
            FROM drop_table_bigrams
            WHERE freq <= {threshold}
        )
        """)

        self(
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
        self(
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
        self(
            """
            ALTER TABLE unigram_db ADD
                hash_index UINT64
            """)
        self(
            """
            UPDATE unigram_db
            SET 
            hash_index = hash(ug)
        """)
        self("DROP TABLE unigram_db_unagg")
        self("DROP TABLE trigram_db_unagg")
        self("VACUUM ANALYZE")
        
    def create_totals(self):
        self(
            """
            CREATE OR REPLACE TABLE trigram_totals AS
            WITH total_freq AS (
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
                FROM total_freq
            ), type_1 AS (
                SELECT max(typef_1) as max_type1_trigram
                FROM (
                    SELECT 
                        ug_3,
                        count( * ) AS typef_1
                    FROM total_freq
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
                    FROM total_freq
                    GROUP BY
                        ug_1,
                        ug_2
                )
            )
            SELECT
                token_frequency.max_token_trigram AS max_token,
                type_1.max_type1_trigram AS max_type1,
                type_2.max_type2_trigram AS max_type2,
                3 as ngram_length
            FROM 
                token_frequency,
                type_1,
                type_2
        """)            

        self(
            """
            CREATE OR REPLACE TABLE bigram_totals AS
            WITH total_freq AS (
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
                FROM total_freq
            ), type_1 AS (
                SELECT 
                    max(typef_1) AS max_type1_bigram
                FROM (
                    SELECT
                        ug_2,
                        count( * ) AS typef_1
                    FROM total_freq
                    GROUP BY ug_2
                )
            ), type_2 AS (
                SELECT max(typef_2) AS max_type2_bigram
                FROM (
                    SELECT ug_1,
                    count( * ) AS typef_2
                FROM total_freq
                GROUP BY ug_1
                )
            )
            SELECT
                token_frequency.max_token_bigram AS max_token,
                type_1.max_type1_bigram AS max_type1,
                type_2.max_type2_bigram AS max_type2,
                2 AS ngram_length
            FROM 
                token_frequency,
                type_1,
                type_2
        """)

        self(
            """
            CREATE OR REPLACE TABLE ngram_totals AS
            SELECT *
            FROM trigram_totals
            UNION ALL 
            SELECT *
            FROM bigram_totals
        """)


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
            """
            CREATE OR REPLACE TABLE ngram_totals AS
            SELECT *
            FROM trigram_totals
            UNION ALL 
            SELECT *
            FROM bigram_totals
        """)

        self(
            """
            CREATE TABLE corpus_proportions AS
                SELECT 
                    corpus,
                    SUM(freq) / (SELECT SUM(freq) FROM unigram_db) AS corpus_prop
                FROM unigram_db 
                GROUP BY corpus
        """)

    def get_unigram(self, unigram):
        unigram_info = self(f"EXECUTE get_unigram({unigram})").fetch_df()
        return unigram_info

# Below: methods related to MWU extraction directly, previously in corpus_helper.

    def make_filtered_db(self, source, target):
            self(
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
            self(
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


    def create_query(self, ngrams, source, target):
        query_df = pd.DataFrame(ngrams, columns = [source, target])
        self.conn.register("query_df", query_df)
        self(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE this_query 
            ({source} TEXT, {target} TEXT)
        """)
        self(
            f"""
            INSERT INTO this_query 
            SELECT {source}, {target} 
            FROM query_df
        """)
        self.make_filtered_db(source, target)


    # MWU functions


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
        # After getting token freq, we can filter non-occurring ngrams from the query
        self.reduce_query(source, target)

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
        f"""
        CREATE OR REPLACE TABLE rel_freqs AS
        WITH source_freq AS (
            SELECT DISTINCT {source},
            freq as source_freq 
            FROM token_freq 
            INNER JOIN (
                SELECT 
                    {source},
                    SUM(freq) AS freq
                FROM filtered_db
                GROUP BY {source}
                )
            USING ({source})
        ), target_freq AS (
            SELECT DISTINCT {target},
            freq as target_freq 
            FROM token_freq 
            INNER JOIN (
                SELECT 
                    {target},
                    SUM(freq) AS freq
                FROM filtered_db
                GROUP BY {target}
                )
            USING ({target})
        )
        SELECT 
            *,
            (SELECT SUM(freq) AS total_freq FROM trigram_db) AS total_freq
        FROM token_freq
        LEFT JOIN source_freq
        USING ({source})
        LEFT JOIN target_freq
        USING ({target})
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
                source_freq / total_freq AS prob_1,
                target_freq / total_freq AS prob_2
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