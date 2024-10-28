import numpy as np
import numpy_groupies as npg
import pandas as pd
import nltk
import preprocess_corpus
import gc

# Assumes as input the kind of object outputted by preprocess_corpus.py (i.e, dictionary of corpus_id: list of sentences where each sentence is a list of unigrams)

# def get_ngrams(corpus_dictionary):
#     ngram_dict = {corpus: [list(nltk.everygrams(sentence, min_len=1, max_len=2)) for sentence in corpus_sents] for corpus, corpus_sents in corpus_dictionary.items()}
#     all_unigrams = {corpus_name: [gram for sent in corpus for gram in sent if len(gram) == 1] for corpus_name, corpus in ngram_dict.items()}
#     all_bigrams = {corpus_name: [gram for sent in corpus for gram in sent if len(gram) == 2] for corpus_name, corpus in ngram_dict.items()}
#     return (all_unigrams, all_bigrams)

# def get_frequencies(ngram_dictionary):
#     frequency_pc = {corpus: nltk.FreqDist(grams) for corpus, grams in ngram_dictionary.items()}
#     overall_freq = nltk.FreqDist()
#     for freq_dist in frequency_pc.values():
#         overall_freq.update(freq_dist)
#     return frequency_pc, overall_freq

def filter_ngrams(total_freq, pc_freq, threshold, verbose=False):
    # in-place
    if verbose:
        n_grams_before = len(total_freq)
        print('Filtering...')
    filtered_ngrams = [ngram for ngram, freq in total_freq.items() if freq < threshold]
    for ngram in filtered_ngrams:
        total_freq.pop(ngram)
        for corpus_dict in pc_freq.values():
            corpus_dict.pop(ngram, None)
    import gc
    if verbose:
        n_grams_after = len(total_freq)
        print(f'Ngrams removed: {n_grams_before - n_grams_after}')
        print('Garbage collecting...')
    gc.collect()

def get_ids(unigram_freqs, bigram_freqs):
    uq_unigrams = [gram for gram in sorted(unigram_freqs.keys())]
    unigram_id = range(len(uq_unigrams))
    unigram_id = dict(zip(uq_unigrams, unigram_id))
    uq_bigrams = sorted(bigram_freqs.keys())
    uq_bigrams = list(uq_bigrams)
    bigram_id = range(len(uq_bigrams))
    bigram_id = dict(zip(uq_bigrams, bigram_id))
    return unigram_id, bigram_id

def get_bigram_info(bigram_freqs, unigram_ids, bigram_ids, by_corpus=True):
    bigram_info = []

    if by_corpus:
        dt = {'names':['first', 'second', 'first_id', 'second_id', 'corpus', 'freq', 'bigram_id'], 'formats':[np.dtypes.StrDType, np.dtypes.StrDType, 'i', 'i',  np.dtypes.StrDType, 'i', 'i']}
    else:
        dt = {'names':['first', 'second', 'first_id', 'second_id', 'freq', 'bigram_id'], 'formats':[np.dtypes.StrDType, np.dtypes.StrDType, 'i', 'i', 'i', 'i']}

    if by_corpus:
        for corpus, grams in bigram_freqs.items():
            for gram in grams.items():
                bigram_info.append((gram[0][0], gram[0][1], unigram_ids[gram[0][0]], unigram_ids[gram[0][1]], corpus, gram[1], bigram_ids[gram[0]]))
    else:
        for gram in bigram_freqs.items():
            bigram_info.append((gram[0][0], gram[0][1], unigram_ids[gram[0][0]], unigram_ids[gram[0][1]], gram[1], bigram_ids[gram[0]]))

    bigram_freqs_array = np.sort(np.array(bigram_info, dtype=dt), axis=0, order='bigram_id')
    
    return bigram_freqs_array

def get_corpus_props(unigram_freqs_pc):
    corpus_sizes = {corpus: dist.total() for corpus, dist in unigram_freqs_pc.items()}
    corpus_total = np.sum(list(corpus_sizes.values()))    
    corpus_props = [(corpus, size / corpus_total) for corpus, size in corpus_sizes.items()]
    corpus_props = pd.DataFrame(corpus_props, columns=['corpus', 'corpus_prop'])
    return corpus_props


# TODO: allow to skip process corpus by allocating

def process_corpus(corpus='bnc', corpus_dir=None, verbose=False):
    # should make brown the default corpus because it's included in nltk
    print('Preprocessing the corpus')
    if corpus == 'bnc' and corpus_dir:
        frequency_dists = preprocess_corpus.preprocess_bnc(corpus_dir, verbose=verbose)
    
    global bigram_per_corpus, bigram_total, corpus_proportions, unigram_frequencies_pc
    
    bigram_per_corpus = frequency_dists['bigram'][0] # These could be generators used only by get_bigram_info. Save RAM until allocating arrays. Can filter on callout from generator maybe.
    bigram_total = frequency_dists['bigram'][1]
    unigram_frequencies_pc = frequency_dists['unigram'][0]
    unigram_total = frequency_dists['unigram'][1]
    
    print('Cleaning the ngrams...')
    
    print('Getting everything ready for score extraction')
    unigram_id, bigram_id = get_ids(unigram_total, bigram_total)
    bigram_per_corpus = get_bigram_info(bigram_per_corpus, unigram_id, bigram_id, by_corpus=True)
    bigram_total = get_bigram_info(bigram_total, unigram_id, bigram_id, by_corpus=False)
    corpus_proportions = get_corpus_props(unigram_frequencies_pc)