import mwu_measures
import pandas as pd
import numpy as np
from line_profiler import LineProfiler
print('its the new one')

bnc_corpus = mwu_measures.process_corpus('bnc', 'bnc_tokenized.txt', chunk_size=1000000, verbose=True)
mwu_examples = pd.read_csv('MultiwordExpression_Concreteness_Ratings.csv')
mwu_examples['length'] = mwu_examples['Expression'].apply(lambda x: len(x.split()))
mwu_examples = mwu_examples.loc[(mwu_examples['length'] == 2) | (mwu_examples['length'] == 3)]
mwu_examples['Expression'] = mwu_examples['Expression'].apply(lambda x: x.lower())

lp = LineProfiler()
lp_wrapper = lp(mwu_measures.mwu_functions.get_ngram_scores)

all_scores = [lp_wrapper(ngram, bnc_corpus) for ngram in mwu_examples.sample(100)['Expression']]


# mwu_scores = lp_wrapper(mwu_examples.sample(1000)['Expression'], bnc_corpus, normalize=True, parallel=False, verbose=True, track_progress=True)
lp.print_stats()
# mwu_scores.to_csv('scratch.csv')