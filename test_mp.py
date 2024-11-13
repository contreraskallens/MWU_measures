import mwu_measures
import pandas as pd
import time

if __name__ == '__main__':
    mwu_measures.process_corpus('bnc', 'bnc_tokenized.txt', chunk_size=1000000, verbose=True)
    mwu_examples = pd.read_csv('MultiwordExpression_Concreteness_Ratings.csv')
    mwu_examples['length'] = mwu_examples['Expression'].apply(lambda x: len(x.split()))
    mwu_examples = mwu_examples.loc[mwu_examples['length'] == 2]
    mwu_examples['Expression'] = mwu_examples['Expression'].apply(lambda x: x.lower())
    mwu_examples = mwu_examples.sample(10000)
    start_time = time.time()
    mwu_scores_single = mwu_measures.get_mwu_scores(mwu_examples['Expression'], normalize=True, track_progress=True, entropy_limits=[-0.05, 0.05])
    single_duration = time.time() - start_time
    start_time = time.time()
    mwu_scores_multi = mwu_measures.get_mwu_scores(mwu_examples['Expression'], normalize=True, parallel=True, ncores=4, entropy_limits=[-0.05, 0.05])
    # file_name = "mwu_measures/cache/fw_dict.json"
    # test_file = mwu_measures.processing_corpus.BIGRAM_FW
    # test_file = {corpus: {ngram_1: dict(freqs) for ngram_1, freqs in ngram_freqs.items()} for corpus, ngram_freqs in test_file.items()}
    # with open(file_name, 'w') as f:
    #     json.dump(test_file, f)
    # with open(file_name, 'r+b') as f:
    #     # Memory-map the entire file
    #     mapped_file = mmap.mmap(f.fileno(), 0)
    
    # Access the data (decoding the JSON data)
    # data = json.loads(mapped_file[:].decode('utf-8'))  # Decode the bytes to a string and load as JSON
    
    multi_duration = time.time() - start_time
    # print(f"JSON time for fw_dict): {multi_duration:.4f} seconds")

    # start_time = time.time()
    # mwu_scores_multi = mwu_measures.get_mwu_scores(mwu_examples['Expression'], normalize=False, parallel=True, ncores=4)
    # file_name = "mwu_measures/cache/fw_dict.pickle"
    # with open(file_name, 'wb') as f:
    #     pickle.dump(test_file, f)

    # # Memory-map the file and load the data
    # with open(file_name, 'r+b') as f:
    #     # Memory-map the file
    #     mapped_file = mmap.mmap(f.fileno(), 0)

    #     # Load the pickled data from memory
    #     data = pickle.loads(mapped_file)    
    # multi_duration = time.time() - start_time
    # print(f"Pickle time for fw_dict): {multi_duration:.4f} seconds")

    print(f"Single-processing time: {single_duration:.4f} seconds")
    print(f"Multiprocessing time processes): {multi_duration:.4f} seconds")

    # print(mwu_scores_single)
    # print(mwu_scores_multi)   

    # mwu_measures.process_corpus(test_corpus=True)
    # x = mwu_measures.get_mwu_scores(['b d', 'c b', 'a c'])
    # x['dispersion'] = 1 - x['dispersion']
    # print(x)

    # y = mwu_measures.get_mwu_scores(['b d', 'c b', 'a c'], parallel=True)
    # y['dispersion'] = 1 - y['dispersion']
    # print(y)