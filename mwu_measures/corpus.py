import duckdb
import pandas as pd
import os
from rich.progress import Progress, TextColumn, SpinnerColumn, MofNCompleteColumn
from rich.console import Console

class Corpus():
    def __init__(self, corpus_name):
        self.corpus_name = corpus_name
        self.path = f"mwu_measures/db/{corpus_name}.db"
        if os.path.isfile(f"mwu_measures/db/{corpus_name}.db"):
            print('Using preexisting corpus')
            # self.conn = duckdb.connect()
        else:
            print("DB doesn't exist. Allocate before calculations.")
            with duckdb.connect(self.path) as conn:
                conn.execute(
                    """
                    CREATE TABLE trigram_db_temp (
                        corpus TEXT,
                        ug_1 UINT64,
                        ug_2 UINT64,
                        ug_3 UINT64,
                        big_1 UINT64,
                        freq INTEGER
                        )
                """)
                conn.execute(
                    """
                    CREATE TABLE unigram_db_temp (
                        corpus TEXT,
                        ug TEXT,
                        ug_hash UINT64,
                        freq INTEGER
                        )
                """)            

    def __call__(self, query):
        with duckdb.connect(self.path) as conn:
            query = conn.execute(query)
            return query.fetchall()
    def df(self, query):
        with duckdb.connect(self.path) as conn:
            query = conn.execute(query)
            return query.fetch_df()
    def add_chunk(self, ngram_lists):
        with duckdb.connect(self.path) as conn:
            chunk_unigrams, chunk_trigrams = ngram_lists
            chunk_unigrams = pd.DataFrame(chunk_unigrams, columns=['corpus', 'ug', 'freq'])
            chunk_unigrams['corpus'] = chunk_unigrams['corpus'].astype(str)
            chunk_trigrams = pd.DataFrame(chunk_trigrams, columns=['corpus', 'ug_1', 'ug_2', 'ug_3', 'freq'])
            chunk_trigrams['corpus'] = chunk_trigrams['corpus'].astype(str)
            chunk_trigrams['big_1'] = chunk_trigrams['ug_1'] + ' ' + chunk_trigrams['ug_2']
            conn.register("unigram_df", chunk_unigrams)
            conn.register("trigram_df", chunk_trigrams)
            
            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE chunk_unigrams AS (
                    SELECT 
                        corpus,
                        ug,
                        HASH(ug) as ug_hash,
                        freq
                    FROM unigram_df
                )
            """)
            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE chunk_trigrams AS (
                    SELECT 
                        corpus,
                        HASH(ug_1) AS ug_1,
                        HASH(ug_2) AS ug_2,
                        HASH(ug_3) AS ug_3,
                        HASH(big_1) AS big_1,
                        freq
                    FROM trigram_df
                )
            """)
            conn.execute(
                """
                INSERT INTO trigram_db_temp
                SELECT * from chunk_trigrams
            """)
            conn.execute(
                """
                INSERT INTO unigram_db_temp
                SELECT * from chunk_unigrams
            """)          

    def consolidate_corpus(self, threshold=2):
        with duckdb.connect(self.path) as conn:
            # conn.execute("""CREATE OR REPLACE TABLE trigram_db_temp AS""")
            all_corpora = conn.execute("SELECT DISTINCT corpus FROM trigram_db_temp").fetchall()
            all_corpora = ["'" + str(corpus_list[0]) + "'" for corpus_list in all_corpora]
            all_corpora = ", ".join(all_corpora)
            conn.execute(
                f"""CREATE OR REPLACE TABLE trigram_db AS (
                    SELECT * 
                    FROM trigram_db_temp
                    PIVOT(
                        SUM(freq) FOR corpus IN ({all_corpora})

                    )
                )
            """)
            conn.execute(
                f"""CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT * 
                    FROM unigram_db_temp
                    PIVOT(
                        SUM(freq) FOR corpus IN ({all_corpora})

                    )
                )
            """)

            conn.execute("DROP TABLE trigram_db_temp")
            conn.execute("DROP TABLE unigram_db_temp")
            conn.execute("VACUUM ANALYZE")
            
            # Replace NA with 0
            conn.execute("""
                CREATE OR REPLACE TABLE trigram_db AS (
                    SELECT 
                        ug_1, 
                        ug_2, 
                        ug_3, 
                        big_1,
                        COALESCE(COLUMNS(* EXCLUDE(ug_1, ug_2, ug_3, big_1)), 0)
                    FROM trigram_db
                )""")
            # Add column with total frequency
            conn.execute("""
                CREATE OR REPLACE TABLE trigram_db AS (
                    SELECT *, 
                    LIST_SUM(LIST_VALUE(*COLUMNS(* EXCLUDE (ug_1, ug_2, ug_3, big_1)))) AS freq
                    FROM trigram_db
                )""")
            # Filter on threshold and clean dummy trigrams
            conn.execute(f"""
                CREATE OR REPLACE TABLE trigram_db AS (
                    SELECT *
                    FROM trigram_db
                    WHERE 
                        freq > {threshold}
                        AND ug_1 != HASH('END')
                        AND ug_2 != HASH('END')
                )""")
            # TODO: ARE THERE REPEATED LINES IN TRIGRAM_DB? WHY?
            # Same for unigrams
            conn.execute("""
                CREATE OR REPLACE TABLE unigram_db AS (
                SELECT ug, ug_hash, COALESCE(COLUMNS(* EXCLUDE(ug, ug_hash)), 0)
                FROM unigram_db
            )""")
            conn.execute("""
                CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT 
                        *,
                        LIST_SUM(LIST_VALUE(*COLUMNS(* EXCLUDE (ug, ug_hash))))
                    AS freq 
                    FROM unigram_db
                )""")
            conn.execute(f"""
                CREATE OR REPLACE TABLE unigram_db AS (
                    SELECT *
                    FROM unigram_db
                    WHERE freq > {threshold}
                )""")
            conn.execute("VACUUM ANALYZE")
        
    def create_totals(self):
        with duckdb.connect(self.path) as conn:
            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE trigram_totals AS
                WITH total_freq AS (
                    SELECT 
                        ug_1,
                        ug_2,
                        ug_3,
                        big_1,
                        freq
                    FROM trigram_db
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

            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE bigram_totals AS
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

            conn.execute(
                """
                CREATE OR REPLACE TABLE ngram_totals AS
                SELECT *
                FROM trigram_totals
                UNION ALL 
                SELECT *
                FROM bigram_totals
            """)

            conn.execute("""
                CREATE OR REPLACE TABLE corpus_proportions AS (
                    SELECT SUM(COLUMNS(* EXCLUDE(ug, ug_hash, freq))) / SUM(freq)
                    FROM unigram_db
            )""")

            # self(
            # """
            #     CREATE OR REPLACE TABLE unigram_totals AS
            #     SELECT 
            #         ug_hash AS ug,
            #         freq,
            #         SUM(SUM(freq)) OVER () AS total_ug,
            #     FROM unigram_db
            #     GROUP BY (ug)
            # """)

            # conn.execute(
            #     """
            #     CREATE OR REPLACE TABLE corpus_proportions AS
            #         SELECT 
            #             corpus,
            #             SUM(freq) / (SELECT SUM(freq) FROM unigram_db) AS corpus_prop
            #         FROM unigram_db 
            #         GROUP BY corpus
            # """)

        # def get_unigram(self, unigram):
        #     unigram_info = self(f"EXECUTE get_unigram({unigram})").fetch_df()
        #     return unigram_info

# Below: methods related to MWU extraction directly, previously in corpus_helper.



    def create_query(self, ngrams, source, target):
        query_df = pd.DataFrame(ngrams, columns = [source, target])
        with duckdb.connect(self.path) as conn:
            conn.register("query_df", query_df)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE query_ref 
                ({source} TEXT, {target} TEXT, {source}_hash UINT64, {target}_hash UINT64)
            """)
            conn.execute(
                f"""
                INSERT INTO query_ref 
                SELECT 
                    {source}, 
                    {target}, 
                    HASH({source}) AS {source}_hash,
                    HASH({target}) AS {target}_hash
                FROM query_df
            """)
            conn.execute(f"""
                CREATE OR REPLACE TABLE this_query AS (
                    SELECT
                        {source}_hash AS {source},
                        {target}_hash AS {target}
                FROM query_ref
            )""")


    # MWU functions

    def make_filtered_db(self, source, target):
            self(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS (
                SELECT
                    {source},
                    {target},
                    SUM(COLUMNS(* EXCLUDE(ug_1, ug_2, ug_3, big_1))),
                FROM trigram_db
                WHERE 
                    {source} IN (SELECT {source} FROM reduced_query) OR 
                    {target} IN (SELECT {target} FROM reduced_query)
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
            SUM(freq) AS token_freq
        FROM trigram_db
        SEMI JOIN this_query
        USING({source}, {target})
        GROUP BY({source}, {target})
        ORDER BY {source}

    """)
        # After getting token freq, we can filter non-occurring ngrams from the query

    def reduce_query(self, source, target):
        with duckdb.connect(self.path) as conn:
        # Make a reduced query table that includes only the ngrams that appear in the corpus.
        # This is a substantial memory and runtime save for later operations.
        # Then, clean the token frequency table.
            conn.execute(
            f"""
            CREATE OR REPLACE TABLE reduced_query AS
            SELECT * 
            FROM this_query
            SEMI JOIN token_freq
            USING({source}, {target})
        """)
            conn.execute(
            f"""
            CREATE OR REPLACE TABLE token_freq AS
            SELECT *
            FROM reduced_query
            LEFT JOIN token_freq
            USING({source}, {target})
        """)
        self.make_filtered_db(source, target)
    def make_type_freq(self, source, target):
        with duckdb.connect(self.path) as conn:
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE type_1 AS
            SELECT 
                {target},
                COUNT( * ) AS typef_1
            FROM filtered_db
            GROUP BY
                {target}
        """)
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE type_2 AS
            SELECT 
                {source},
                COUNT( * ) AS typef_2
            FROM filtered_db
            GROUP BY
                {source}
        """)
            # Make it temporary so that the computations happen in memory.
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE type_freq_temp AS
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
            # Then assign to on-disk table.
            conn.execute(
            f"""
            CREATE OR REPLACE TABLE type_freq AS
            SELECT *
            FROM reduced_query
            LEFT JOIN type_freq_temp
            USING({source}, {target})
        """)

    def make_dispersion(self, source, target):
        with duckdb.connect(self.path) as conn:
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE prop_table AS
            SELECT 
                {source},
                {target},
                COLUMNS(* EXCLUDE({source}, {target}, freq)) / freq
            FROM filtered_db
            RIGHT JOIN reduced_query
            USING({source}, {target})
        """)
            corpus_names = conn.execute(
                """
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_name = 'filtered_db'
                    AND column_name NOT IN ('ug_1', 'ug_2', 'ug_3', 'big_1', 'freq')
        """).fetchall()
            corpus_names = [name[0] for name in corpus_names]
            join_query = ', '.join([f'''
                                    CASE 
                                        WHEN prop_table."{name}" > 0 
                                            THEN prop_table."{name}" * log2(prop_table."{name}" / corpus_proportions."{name}") 
                                        ELSE 0 
                                    END AS "{name}"''' for name in corpus_names])
            
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE dispersion_temp AS
            WITH dist_table AS (
                SELECT 
                    {source},
                    {target},
                    {join_query}
                FROM prop_table, corpus_proportions
            ), kld_table AS (
                SELECT
                    {source},
                    {target},
                    LIST_SUM(LIST_VALUE(*COLUMNS(* EXCLUDE ({source}, {target})))) AS kld
                FROM dist_table
            )
            SELECT 
                {source},
                {target},
                1 - pow(EXP(1), -(kld)) as dispersion
            FROM kld_table
        """)

            conn.execute(
                f"""
                CREATE OR REPLACE TABLE dispersion AS
                SELECT * 
                FROM reduced_query
                LEFT JOIN dispersion_temp
                USING({source}, {target})
        """)
    def make_associations(self, source, target):
        with duckdb.connect(self.path) as conn:
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE rel_freqs AS
            WITH source_freq AS (
                SELECT 
                    DISTINCT {source},
                    freq as source_freq 
                FROM token_freq 
                LEFT JOIN (
                    SELECT 
                        {source},
                        SUM(freq) AS freq
                    FROM filtered_db
                    GROUP BY {source}
                    )
                USING ({source})
            ), target_freq AS (
                SELECT 
                    DISTINCT {target},
                    freq as target_freq 
                FROM token_freq 
                LEFT JOIN (
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
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE associations_temp AS
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
            conn.execute(
            f"""
            CREATE OR REPLACE TABLE associations AS
            SELECT *
            FROM reduced_query
            LEFT JOIN associations_temp
            USING({source}, {target})
        """)

    def make_entropy_diff(self, source, target, slot):
        # There has to be a better way of doing all this CTEing for entropies. Functions?
        # The second half of the procedure is the same as the first but with a 
        # different input (0 if query_target == target)
        # Real entropy
        with duckdb.connect(self.path) as conn:
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE entropy_real_{slot} AS
            WITH all_freqs AS (
                SELECT
                    {source},
                    {target},
                    freq AS total_freq
                FROM filtered_db
                SEMI JOIN reduced_query
                USING({source})
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
        """)

    # Counterfactual entropy

            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE entropy_cf_{slot} AS
            WITH all_freqs AS (
                SELECT
                    {source},
                    {target},
                    freq AS total_freq
                FROM filtered_db
                SEMI JOIN reduced_query
                USING({source})
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
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE entropy_{slot}_temp AS
            SELECT 
                {source},
                {target},
                entropy_cf - entropy_real AS entropy_{slot}
            FROM entropy_real_{slot}
            INNER JOIN entropy_cf_{slot}
            USING({source}, {target})
        """)
            conn.execute(f"""
            CREATE OR REPLACE TABLE entropy_{slot} AS
            SELECT *
            FROM entropy_{slot}_temp
        """)
        


    def make_entropy_diffs(self, source, target):
        self.make_entropy_diff(source, target, '2')
        self.make_entropy_diff(target, source, '1')
        with duckdb.connect(self.path) as conn:
            conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE entropy_diffs_temp AS
            SELECT * FROM entropy_1
            INNER JOIN entropy_2
            USING({source}, {target})
        """)
            conn.execute(
            f"""
            CREATE OR REPLACE TABLE entropy_diffs AS
            SELECT 
                {source},
                {target},
                COALESCE(entropy_1, 1) AS entropy_1,
                COALESCE(entropy_2, 1) AS entropy_2
            FROM reduced_query
            LEFT JOIN entropy_diffs_temp
            USING({source}, {target})
        """)
            conn.execute("DROP TABLE entropy_1")
            conn.execute("DROP TABLE entropy_2")
            conn.execute("VACUUM ANALYZE")


    def join_measures(self, source, target, length):
        with duckdb.connect(self.path) as conn:
            conn.execute("ATTACH ':memory:' AS results")
            conn.execute(
                f"""
                CREATE TABLE results.raw_measures_temp AS
                SELECT *, {length} ngram_length 
                FROM reduced_query
            """)
            conn.execute(
                f"""CREATE OR REPLACE TABLE results.raw_measures_temp AS
                SELECT * 
                FROM results.raw_measures_temp
                LEFT JOIN token_freq USING({source}, {target})
            """)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS 
                SELECT * 
                FROM results.raw_measures_temp 
                LEFT JOIN dispersion 
                USING({source}, {target})
                """)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE results.raw_measures_temp 
                AS SELECT *
                FROM results.raw_measures_temp 
                LEFT JOIN type_freq 
                USING({source}, {target})
                """)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS 
                SELECT * 
                FROM results.raw_measures_temp 
                LEFT JOIN entropy_diffs 
                USING({source}, {target})
            """)
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE results.raw_measures_temp AS 
                SELECT * 
                FROM results.raw_measures_temp 
                LEFT JOIN associations 
                USING({source}, {target})
            """)

            conn.execute("DROP TABLE token_freq")
            conn.execute("DROP TABLE dispersion")
            conn.execute("DROP TABLE type_freq")
            conn.execute("DROP TABLE entropy_diffs")
            conn.execute("DROP TABLE associations")
            conn.execute("DROP TABLE reduced_query")
            conn.execute("DROP TABLE filtered_db")
            conn.execute(
                f"""
                CREATE OR REPLACE TABLE raw_measures AS
                SELECT 
                    query_ref.{source}, 
                    query_ref.{target}, 
                    token_freq, 
                    dispersion, 
                    typef_1, 
                    typef_2, 
                    entropy_1, 
                    entropy_2, 
                    fw_assoc, 
                    bw_assoc,
                    ngram_length
                FROM query_ref 
                LEFT JOIN results.raw_measures_temp 
                ON query_ref.{source}_hash = raw_measures_temp.{source} 
                    AND query_ref.{target}_hash = raw_measures_temp.{target}
            """)
            conn.execute("DROP TABLE query_ref")
            conn.execute("VACUUM ANALYZE")

    def normalize_measures(self, source, target):
        with duckdb.connect(self.path) as conn:
            conn.execute(
                """
                CREATE OR REPLACE TEMPORARY TABLE min_max AS 
                SELECT
                    MIN(token_freq) AS min_tok,
                    MAX(token_freq) AS max_tok,
                    MIN(typef_1) AS min_type_1,
                    MAX(typef_1) AS max_type_1,
                    MIN(typef_2) AS min_type_2,
                    MAX(typef_2) AS max_type_2,
                    MIN(entropy_1) AS min_entropy_1,
                    MAX(entropy_1) AS max_entropy_1,
                    MIN(entropy_2) AS min_entropy_2,
                    MAX(entropy_2) AS max_entropy_2,
                    MIN(fw_assoc) AS min_fw_assoc,
                    MAX(fw_assoc) AS max_fw_assoc,
                    MIN(bw_assoc) AS min_bw_assoc,
                    MAX(bw_assoc) AS max_bw_assoc,
                    ngram_length
                FROM raw_measures
                WHERE NOT ISNAN(ngram_length)
                GROUP BY ngram_length
                """
                )

            conn.execute(
                f"""
                CREATE OR REPLACE TEMPORARY TABLE normalized_measures_temp AS
                SELECT
                    {source},
                    {target},
                    (LOG(token_freq) - LOG(min_max.min_tok)) / (LOG(min_max.max_tok) - LOG(min_max.min_tok)) AS token_freq,
                    1 - dispersion AS dispersion,
                    1 - ((LOG(typef_1) - LOG(min_max.min_type_1)) / (LOG(min_max.max_type_1) / LOG(min_max.min_type_1))) AS type_1,
                    1 - ((LOG(typef_2) - LOG(min_max.min_type_2)) / (LOG(min_max.max_type_2) / LOG(min_max.min_type_2))) AS type_2,
                    (entropy_1 - min_max.min_entropy_1) / (min_max.max_entropy_1 - min_max.min_entropy_1) AS entropy_1,
                    (entropy_2 - min_max.min_entropy_2) / (min_max.max_entropy_2 - min_max.min_entropy_2) AS entropy_2,
                    (fw_assoc - min_max.min_fw_assoc) / (min_max.max_fw_assoc - min_max.min_fw_assoc) AS fw_assoc,
                    (bw_assoc - min_max.min_bw_assoc) / (min_max.max_bw_assoc - min_max.min_bw_assoc) AS bw_assoc,
                    ngram_length
            FROM raw_measures
            LEFT JOIN min_max
            USING (ngram_length)
        """)
            conn.execute(
                """
                CREATE OR REPLACE TABLE normalized_measures AS 
                SELECT * 
                FROM normalized_measures_temp
        """)

    def get_ngram_scores(self, source, target, length, entropy_limits=[-0.1, 0.1]):
        with Progress(
            TextColumn("[bold blue]{task.fields[task_name]}\n[bold blue]{task.description}", justify="left"),
            MofNCompleteColumn(),
            SpinnerColumn(),
            transient=True) as progress:
            compute_mwus = progress.add_task("Initializing...", task_name=f"Computing MWU scores, length {length}", total=8)
            progress.update(compute_mwus, advance=1, description="Token frequencies...")
            self.make_token_freq(source, target)
            self.reduce_query(source, target)
            progress.update(compute_mwus, advance=1, description="Type frequencies...")
            self.make_type_freq(source, target)
            progress.update(compute_mwus, advance=1, description="Dispersion...")
            self.make_dispersion(source, target)
            progress.update(compute_mwus, advance=1, description="Association...")
            self.make_associations(source, target)
            progress.update(compute_mwus, advance=1, description="Entropy differences...")
            self.make_entropy_diffs(source, target)
            progress.update(compute_mwus, advance=1, description="Joining everything...")
            self.join_measures(source, target, length)
            progress.update(compute_mwus, advance=1, description="Normalizing...")
            self.normalize_measures(source, target)

            progress.update(compute_mwus, advance=1, description="Cleaning up...")
            raw_measures = self.df("SELECT * FROM raw_measures")
            normalized_measures = self.df("SELECT * FROM normalized_measures")
            self("DROP TABLE this_query")
            self("DROP TABLE raw_measures")
            self("DROP TABLE normalized_measures")
            progress.update(compute_mwus, advance=1, description="Done!")

            return {
                'raw': raw_measures,
                'normalized': normalized_measures,
            }