import pandas as pd
from orjson import loads
import numpy as np
from functools import reduce

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

    def create_bigram_query(self, ngrams):
        query_df = pd.DataFrame((ngram.split() for ngram in ngrams), columns = ['ug_1', 'ug_2'])
        self.conn.execute(
            """
            CREATE OR REPLACE TEMPORARY TABLE this_query 
            (ug_1 TEXT, ug_2 TEXT)
        """)
        self.conn.execute("""
            INSERT INTO this_query SELECT ug_1, ug_2 FROM query_df
        """)
    def allocate_filtered_db(self):
            self.conn.execute(
            """
            CREATE OR REPLACE TABLE filtered_db AS (
                SELECT
                    corpus,
                    ug_1,
                    ug_2,
                    SUM(freq) AS freq,
                FROM trigram_db
                WHERE 
                    ug_1 IN (SELECT ug_1 FROM this_query) OR 
                    ug_2 IN (SELECT ug_2 FROM this_query)
                GROUP BY 
                    corpus,
                    ug_1,
                    ug_2
            )
        """)
            self.conn.execute(
                """
                CREATE OR REPLACE TABLE filtered_db_total AS (
                    SELECT
                        ug_1,
                        ug_2,
                        SUM(freq) AS total_freq,
                    FROM filtered_db
                    GROUP BY 
                        ug_1,
                        ug_2
                )
            """)


    def allocate_for_bigrams(self, source, target):
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE filtered_db AS
            SELECT
                corpus,
                ug_1,
                ug_2,
                CONCAT_WS(' ', ug_1, ug_2) AS ngram,
                freq
            FROM trigram_db
            WHERE
                ngram NOT IN (SELECT ngram FROM drop_table_bigrams)
        """)
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS
            SELECT
                corpus,
                hash({source}) as hash_index,
                {target},
                SUM(freq) AS freq
            FROM filtered_db
            WHERE hash_index IN (
                SELECT hash_index FROM this_query
                ) 
            GROUP BY
                corpus,
                hash_index,
                {target}
        """)
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE joined_db AS                  
            SELECT 
                query,
                corpus,
                ug_1,
                ug_2,
                SUM(freq) as freq
            FROM this_query
            INNER JOIN filtered_db
            USING(hash_index)
            GROUP BY
                query,
                corpus,
                ug_1,
                ug_2
            ORDER BY {source}, freq DESC
        """)
    def allocate_for_trigrams(self, source, target):
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS
            SELECT
                corpus,
                CONCAT_WS(' ', ug_1, ug_2) AS ngram,
                hash(ngram) AS ngram_hash,
                hash({source}) as hash_index,
                {target},
                freq
            FROM trigram_db
            WHERE 
                hash_index IN (
                    SELECT hash_index FROM this_query
                ) AND
                ngram_hash NOT IN (SELECT ngram_hash FROM drop_table_trigrams)
        """)
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE joined_db AS                  
            SELECT 
                query,
                corpus,
                this_query.{source},
                filtered_db.{target},
                freq
            FROM this_query
            INNER JOIN filtered_db
            USING(hash_index)
        """)
        self.conn.execute("DROP TABLE filtered_db")
        self.conn.execute("VACUUM ANALYZE")

    def pack_query(self, target):
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE dist_table AS 
            SELECT 
                query,
                corpus,
                json_group_object({target}, freq) AS freq_dist
            FROM joined_db
            GROUP BY 
                query,
                corpus
        """)
        self.conn.execute("DROP TABLE joined_db")
        self.conn.execute("VACUUM ANALYZE")
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE corpus_db AS
            SELECT
                query,
                json_group_object(corpus, freq_dist) as corpus_array
            FROM dist_table
            GROUP BY query
        """)
        self.conn.execute("DROP TABLE dist_table")
        self.conn.execute("VACUUM ANALYZE")
        result = self.conn.execute(
            """
            SELECT json_group_object(query, corpus_array)
            FROM corpus_db
        """)
        result = result.fetchone()[0] # gets loaded as a tuple with only 1 element
        self.conn.execute("DROP TABLE corpus_db")
        self.conn.execute("VACUUM ANALYZE")
        return self.process_json_result(result)

    def process_json_result(self, json_result):
        result = loads(json_result)
        return result

    def fetch_probabilities(self, ngrams, mode, direction):
        if direction == 'fw':
            if mode == 'bigram':
                source = 'ug_1'
                target = 'ug_2'
            elif mode == 'trigram':
                source = 'big_1'
                target = 'ug_3'
        elif direction == 'bw':
            if mode == 'bigram':
                source = 'ug_2'
                target = 'ug_1'
            elif mode == 'trigram':
                source = 'ug_3'
                target = 'big_1'

        self.allocate_query(source, ngrams)
        if mode == 'bigram':
            self.allocate_for_bigrams(source, target)
        elif mode == 'trigram':
            self.allocate_for_trigrams(source, target)
        
        query_result = self.pack_query(target)

        return query_result

    def run_ngrams(self, ngrams):
        all_ngrams = [ngram.split() for ngram in ngrams]
        bigrams_fw = []
        bigrams_bw = []
        trigrams_fw = []
        trigrams_bw = []
        for ngram in all_ngrams:
            if len(ngram) == 2:
                bigrams_fw.append([' '.join(ngram), ngram[0]])
                bigrams_bw.append([' '.join(ngram), ngram[1]])
            elif len(ngram) == 3:
                trigrams_fw.append([' '.join(ngram), ' '.join([ngram[0], ngram[1]])])
                trigrams_bw.append([' '.join(ngram), ngram[2]])

        bigrams_fw = self.fetch_probabilities(bigrams_fw, 'bigram', 'fw')
        bigrams_bw = self.fetch_probabilities(bigrams_bw, 'bigram', 'bw')
        trigrams_fw = self.fetch_probabilities(trigrams_fw, 'trigram', 'fw')
        trigrams_bw = self.fetch_probabilities(trigrams_bw, 'trigram', 'bw')
        
        forward_probs = bigrams_fw | trigrams_fw
        backward_probs = bigrams_bw | trigrams_bw

        return{'fw': forward_probs, 'bw': backward_probs}

    def get_scores(self, ngrams, chunks):
        ngram_chunks = np.array_split(ngrams, chunks)
        all_results = [self.run_ngrams(ngram_selection) for ngram_selection in ngram_chunks]
        all_results = reduce(lambda x, y: {'fw': x['fw'] | y['fw'], 'bw': x['bw'] | y['bw']}, all_results)
        return all_results