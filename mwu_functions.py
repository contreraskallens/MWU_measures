import numpy as np
import compute_functions
import process_corpus
import pandas as pd
import numpy_groupies as npg


def get_dispersion(bigram_freq_pc, corpus_props):
    bigram_props = bigram_freq_pc['freq'] / np.sum(bigram_freq_pc['freq'])
    bigram_props = zip(bigram_freq_pc['corpus'], bigram_props)
    bigram_props = pd.DataFrame(bigram_props, columns=['corpus', 'ngram_prop'])
    bigram_props = pd.merge(corpus_props, bigram_props, on='corpus', how='left').fillna(0)
    KLD_props = compute_functions.get_KLD(bigram_props['ngram_prop'].values, bigram_props['corpus_prop'].values)
    return KLD_props

def get_entropy_dif(ngram_1_freqs, ngram_2, slot):
    freqs_1 = ngram_1_freqs['freq']
    entropy = compute_functions.get_entropy(freqs_1)
    freqs_1_no2 = freqs_1[ngram_1_freqs[slot] != ngram_2]
    entropy_cf = compute_functions.get_entropy(freqs_1_no2)
    h_diff = entropy_cf - entropy
    return h_diff

def get_association(comp_1, comp_2, token_freq, unigram_frequencies):
    # joint probability is conditioned on the unigram frequencies
    prob_1_2 = token_freq / unigram_frequencies[(comp_1,)]
    prob_2_1 = token_freq / unigram_frequencies[(comp_2,)]
    prob_1 = unigram_frequencies.freq((comp_1,))
    prob_2 = unigram_frequencies.freq((comp_2,))
    assoc_f = compute_functions.get_KLD(np.array([prob_1_2, 1 - prob_1_2]), np.array([prob_2, 1 - prob_2]))
    assoc_b = compute_functions.get_KLD(np.array([prob_2_1, 1 - prob_2_1]), np.array([prob_1, 1 - prob_1]))
    return assoc_f, assoc_b

def get_bigram_scores(comp_1, comp_2, bigram_freqs_per_corpus, corpus_props, bigram_freqs_total, unigram_frequencies):
    bigram_freq = bigram_freqs_per_corpus[np.logical_and(bigram_freqs_per_corpus['first'] == comp_1, bigram_freqs_per_corpus['second'] == comp_2)]
    
    # Token frequency
    token_freq = np.sum(bigram_freq['freq'])

    # Dispersion
    dispersion = get_dispersion(bigram_freq, corpus_props)
    
    ## Need aggregate frequencies for the rest
    freqs_ngram_1 = bigram_freqs_total[bigram_freqs_total['first'] == comp_1]
    freqs_ngram_2 = bigram_freqs_total[bigram_freqs_total['second'] == comp_2]

    # Type frequencies    
    typef_1 = freqs_ngram_2.shape[0]
    typef_2 = freqs_ngram_1.shape[0] # Type frequency of slot 1 is the number of rows that have slot 1 as first

    # Entropy
    slot1_diff = get_entropy_dif(freqs_ngram_2, comp_1, 'first')
    slot2_diff = get_entropy_dif(freqs_ngram_1, comp_2, 'second')

    # Association
    assoc_f, assoc_b = get_association(comp_1, comp_2, token_freq, unigram_frequencies)

    return {'ngram': (comp_1, comp_2), 'first': comp_1, 'second': comp_2, 'token_freq': token_freq, 'dispersion': dispersion, 'type_1': typef_1, 'type_2': typef_2, 'entropy_1': slot1_diff, 'entropy_2': slot2_diff, 'assoc_f': assoc_f, 'assoc_b': assoc_b}


def normalize_scores(bigram_scores, bigram_freqs, entropy_limits=None, scale_entropy=False):
    # Token: min_max
    normalized_scores = bigram_scores.copy()
    log_token_freqs = np.log(bigram_freqs['freq'])
    min_token = np.min(log_token_freqs)
    max_token = np.max(log_token_freqs)
    log_token_target = np.log(bigram_scores['token_freq'])
    normalized_scores['token_freq'] = compute_functions.min_max_norm(log_token_target, min_token, max_token)

    # Dispersion: substract from 1
    normalized_scores['dispersion'] = 1 - normalized_scores['dispersion']

    # Types: min_max and substract from 1
    type_1 = npg.aggregate(bigram_freqs['first_id'], 1)
    type_1 = type_1[bigram_freqs['first_id']]
    type_2 = npg.aggregate(bigram_freqs['second_id'], 1) 
    type_2 = type_2[bigram_freqs['second_id']]
    log_type_1 = np.log(type_1)
    log_type_2 = np.log(type_2)
    min_type_1 = np.min(log_type_1)
    max_type_1 = np.max(log_type_1)
    min_type_2 = np.min(log_type_2)
    max_type_2 = np.max(log_type_2)
    log_type_1_target = np.log(bigram_scores['type_1'])
    log_type_2_target = np.log(bigram_scores['type_2'])
    normalized_scores['type_1'] = 1 - compute_functions.min_max_norm(log_type_1_target, min_type_1, max_type_1)
    normalized_scores['type_2'] = 1 - compute_functions.min_max_norm(log_type_2_target, min_type_2, max_type_2)

    # Entropy, min-max normalize. Optional: limit the range and spread values with cubic root.
    entropy_1 = bigram_scores['entropy_1']
    entropy_2 = bigram_scores['entropy_2']    
    
    if entropy_limits:
        if entropy_1 < entropy_limits[0]:
            entropy_1 = entropy_limits[0]
        elif entropy_1 > entropy_limits[1]:
            entropy_1 = entropy_limits[1]
        if entropy_2 < entropy_limits[0]:
            entropy_2 = entropy_limits[0]
        elif entropy_2 > entropy_limits[1]:
            entropy_2 = entropy_limits[1]

    if scale_entropy:
        entropy_limits[0] = np.cbrt(entropy_limits[0])
        entropy_limits[1] = np.cbrt(entropy_limits[1])
        entropy_1 = np.cbrt(entropy_1)
        entropy_2 = np.cbrt(entropy_2)
    
    if entropy_limits:
        entropy_1 = compute_functions.min_max_norm(entropy_1, entropy_limits[0], entropy_limits[1])            
        entropy_2 = compute_functions.min_max_norm(entropy_2, entropy_limits[0], entropy_limits[1])
    else:
        entropy_1 = compute_functions.min_max_norm(entropy_1, -1, 1)
        entropy_2 = compute_functions.min_max_norm(entropy_2, -1, 1)

    normalized_scores['entropy_1'] = entropy_1
    normalized_scores['entropy_2'] = entropy_2

    return normalized_scores


def test():
    print(process_corpus.unigram_frequencies_pc)
# TODO: make these refer to the global variables in process_corpus
# def get_mwu_scores(ngrams):
#     all_scores = []
#     i = 0
#     for ngram in ngrams:
#         print(ngram)
#         print(i)
#         comps = ngram.split(' ')
#         if len(comps) == 2:
#             bigram_scores = get_bigram_scores(comps[0], comps[1])
#             all_scores.append(bigram_scores)
#         else:
#             bigram_1_score = get_bigram_scores(comps[0], comps[1])
#             bigram_2_score = get_bigram_scores(comps[1], comps[2])
#             all_scores.append(bigram_1_score)
#             all_scores.append(bigram_2_score)
#         i += 1
#     return pd.DataFrame(all_scores)