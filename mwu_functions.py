import numpy as np
import compute_functions

def get_bigram_scores(comp_1, comp_2, bigram_freqs_per_corpus, corpus_props, bigram_freqs_total, unigram_frequencies):
    bigram_freq = bigram_freqs_per_corpus[np.logical_and(bigram_freqs_per_corpus['first'] == comp_1, bigram_freqs_per_corpus['second'] == comp_2)]
    bigram_props = bigram_freq['freq'] / np.sum(bigram_freq['freq'])
    
    # Token frequency
    token_freq = np.sum(bigram_freq['freq'])

    # Dispersion
    ngram_corpus_props = np.zeros(corpus_props.shape[0])
    ngram_corpus_props[bigram_freq['corpus']] = bigram_props # This handles dispersion with bigrams with 0 occurrence in 1 or more corpora
    dispersion = compute_functions.get_KLD(ngram_corpus_props, corpus_props)
    
    ## Need aggregate frequencies for the rest
    freqs_ngram_1 = bigram_freqs_total[bigram_freqs_total['first'] == comp_1]
    freqs_ngram_2 = bigram_freqs_total[bigram_freqs_total['second'] == comp_2]

    # Type frequencies    
    typef_1 = freqs_ngram_2.shape[0]
    typef_2 = freqs_ngram_1.shape[0] # Type frequency of slot 1 is the number of rows that have slot 1 as first

    # Entropies

    slot1_diff = compute_functions.get_entropy_dif(freqs_ngram_2, comp_1, 'first')
    slot2_diff = compute_functions.get_entropy_dif(freqs_ngram_1, comp_2, 'second')

    # Associations
    # joint probability is conditioned on the unigram frequency of the 1st one
    prob_1_2 = token_freq / unigram_frequencies[(comp_1,)]
    prob_2_1 = token_freq / unigram_frequencies[(comp_2,)]
    prob_1 = unigram_frequencies.freq((comp_1,))
    prob_2 = unigram_frequencies.freq((comp_2,))
    assoc_f = compute_functions.get_KLD(np.array([prob_1_2, 1 - prob_1_2]), np.array([prob_2, 1 - prob_2]))
    assoc_b = compute_functions.get_KLD(np.array([prob_2_1, 1 - prob_2_1]), np.array([prob_1, 1 - prob_1]))

    return {'ngram': (comp_1, comp_2), 'first': comp_1, 'second': comp_2, 'token_freq': token_freq, 'dispersion': dispersion, 'type_1': typef_1, 'type_2': typef_2, 'entropy_1': slot1_diff, 'entropy_2': slot2_diff, 'assoc_f': assoc_f, 'assoc_b': assoc_b}