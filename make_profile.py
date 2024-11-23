import mwu_measures
import pandas as pd
import numpy as np
from line_profiler import LineProfiler
import orjson
import mwu_measures.processing_corpus

mwu_examples = pd.read_csv('MultiwordExpression_Concreteness_Ratings.csv')
mwu_examples['length'] = mwu_examples['Expression'].apply(lambda x: len(x.split()))
mwu_examples = mwu_examples.loc[(mwu_examples['length'] == 2) | (mwu_examples['length'] == 3)]
mwu_examples['Expression'] = mwu_examples['Expression'].apply(lambda x: x.lower())

this_corpus = mwu_measures.processing_corpus.process_corpus('bnc', 'bnc_tokenized.txt', chunk_size=1000000, verbose=True)

ngram_selection = mwu_examples['Expression'].tolist()
ngram_chunks = np.array_split(ngram_selection, 3)

def get_all_dists_filter_before(ngrams):
    this_corpus.corpus_conn.execute(
        """
        CREATE OR REPLACE TEMPORARY TABLE bigram_fw_query (query TEXT, ug_1 TEXT)
    """)
    all_ngrams = [ngram.split() for ngram in ngrams]
    bigrams_fw = [[' '.join(ngram), ngram[0]] for ngram in all_ngrams if len(ngram) == 2]
    this_corpus.corpus_conn.executemany("INSERT INTO bigram_fw_query VALUES (?, ?)", bigrams_fw)
    this_corpus.corpus_conn.execute("CREATE OR REPLACE TABLE this_query AS (SELECT query, ug_1, hash(ug_1) as hash_index FROM bigram_fw_query)")
    this_corpus.corpus_conn.execute(
        """
        CREATE OR REPLACE TABLE filtered_db AS
        SELECT
            corpus,
            hash_index,
            ug_2,
            SUM(freq) AS freq
        FROM trigram_db
        WHERE hash_index IN (
            SELECT hash_index FROM this_query
            )
        GROUP BY
            corpus,
            hash_index,
            ug_2
    """)
    this_corpus.corpus_conn.execute(
        """
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
    """)
    this_corpus.corpus_conn.execute(
        """
        CREATE OR REPLACE TABLE dist_table AS 
        SELECT 
            query,
            corpus,
            json_group_object(ug_2, freq) AS freq_dist
        FROM joined_db
        GROUP BY 
            query,
            corpus
    """)
    this_corpus.corpus_conn.execute(
        """
        CREATE OR REPLACE TABLE corpus_db AS
        SELECT
            query,
            json_group_object(corpus, freq_dist) as corpus_array
        FROM dist_table
        GROUP BY query
    """)
    bigrams_fw = this_corpus.corpus_conn.execute(
        """
        SELECT json_group_object(query, corpus_array)
        FROM corpus_db
    """)
    
    print('Fetching...')
    bigrams_fw = bigrams_fw.fetchone()[0] # gets loaded as a tuple with only 1 element
    print('Organizing...')
    bigrams_fw = orjson.loads(bigrams_fw)
    print(f'{len(bigrams_fw)} bigrams in chunk')

this_corpus.corpus_conn.execute("ALTER TABLE trigram_db ADD COLUMN hash_index UINT64")
this_corpus.corpus_conn.execute("UPDATE trigram_db SET hash_index = hash(ug_1)")

lp = LineProfiler()
print('Filter and then aggregate...')
i = 0
lp_wrapper_filter_before = lp(get_all_dists_filter_before)

# lp_wrapper_filter_before(ngram_selection)

for chunk in ngram_chunks:
    print(i)
    lp_wrapper_filter_before(chunk)
    i += 1
lp.print_stats()

lp_no_filter = LineProfiler()
print('No filtering...')
i = 0
lp_wrapper_no_filter = lp_no_filter(get_all_dists_no_filter)

# lp_wrapper_filter_before(ngram_selection)
for chunk in ngram_chunks:
    print(i)
    lp_wrapper_no_filter(chunk)
    i += 1
lp_no_filter.print_stats()