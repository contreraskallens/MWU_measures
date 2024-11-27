import pandas as pd
from orjson import loads

class Fetcher():
    def __init__(self, conn):
        self.conn = conn


    def process_json_result(self, json_result):
        result = json_result.fetchone()[0] # gets loaded as a tuple with only 1 element
        result = loads(result)
        return result



    def allocate_query(self, source, ngrams):
        query_df = pd.DataFrame(ngrams, columns = ['query', source])
        self.conn.execute(
            f"""
            CREATE OR REPLACE TEMPORARY TABLE this_query (query TEXT, {source} TEXT, hash_index UINT64)
        """)
        self.conn.execute(f"""
            INSERT INTO this_query SELECT *, hash({source}) as hash_index FROM query_df
        """)

    def allocate_for_bigrams(self, source, target):
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS
            SELECT
                corpus,
                hash({source}) as hash_index,
                {target},
                SUM(freq) AS freq
            FROM trigram_db
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
    def allocate_for_trigrams(self, source, target, query_df):
        self.conn.execute(
            f"""
            CREATE OR REPLACE TABLE filtered_db AS
            SELECT
                corpus,
                hash({source}) as hash_index,
                {target},
                freq
            FROM trigram_db
            WHERE 
            hash_index IN (
                SELECT hash_index FROM this_query
            )
        """)
        self.execute(
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
        self.conn.execute(
            """
            CREATE OR REPLACE TABLE corpus_db AS
            SELECT
                query,
                json_group_object(corpus, freq_dist) as corpus_array
            FROM dist_table
            GROUP BY query
        """)
        result = self.conn.execute(
            """
            SELECT json_group_object(query, corpus_array)
            FROM corpus_db
        """)
        return self.process_json_result(result)

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
                trigrams_fw = [' '.join(ngram), ' '.join([ngram[0], ngram[1]])]
                trigrams_bw = [' '.join(ngram), ngram[2]]

        # for each one: allocate_query, allocate_for_bigram/trigram, pack_query. Return {'bigram': 'fw", 'bw'} etc.

        # if direction == 'fw':
        #     bigrams = [[' '.join(ngram), ngram[0]] for ngram in all_ngrams if len(ngram) == 2]
        #     source = 'ug_1'
        #     target = 'ug_2'
        # elif direction == 'bw':
        #     bigrams = [[' '.join(ngram), ngram[1]] for ngram in all_ngrams if len(ngram) == 2]
        #     source = 'ug_2'
        #     target = 'ug_1'
        
        # return bigrams

    
