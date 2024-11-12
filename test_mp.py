import mwu_measures
import pandas as pd
import time
import joblib

if __name__ == '__main__':
    mwu_measures.process_corpus('bnc', 'small_corpus.txt', chunk_size=100000, verbose=True)
    mwu_examples = pd.read_csv('MultiwordExpression_Concreteness_Ratings.csv')
    mwu_examples['length'] = mwu_examples['Expression'].apply(lambda x: len(x.split()))
    mwu_examples = mwu_examples.loc[mwu_examples['length'] == 2]
    mwu_examples['Expression'] = mwu_examples['Expression'].apply(lambda x: x.lower())

    start_time = time.time()
    mwu_scores_single = mwu_measures.get_mwu_scores(mwu_examples['Expression'], normalize=False, track_progress=True)
    single_duration = time.time() - start_time
    print(f'Number of cores: {joblib.cpu_count()}')
    start_time = time.time()
    mwu_scores_multi = mwu_measures.get_mwu_scores(mwu_examples['Expression'], normalize=False, parallel=True, ncores=joblib.cpu_count() - 1)
    multi_duration = time.time() - start_time

    print(f"Single-processing time: {single_duration:.4f} seconds")
    print(f"Multiprocessing time processes): {multi_duration:.4f} seconds")

    # print(mwu_scores_single)
    # print(mwu_scores_multi)   