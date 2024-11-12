"""
Main module for obtaining MWU scores. Contains functions for 
obtaining each measure and the main function for processing a list of ngrams.
"""

from collections import defaultdict
from nltk import FreqDist
import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from joblib import Parallel, delayed, parallel_config, dump, load
import os
# import dask
# dask.config.set(scheduler='processes') 

from .compute_functions import min_max_norm
from . import compute_functions
from . import processing_corpus

CONSOLIDATED_FW = None
CONSOLIDATED_BW = None

def get_dispersion(bigram_freq, token_freq, corpus_proportions):
    """
    Computes the "Dispersion" variable for an ngram as the
    KLD divergence between its occurrences in each corpus and
    the overall corpus proportions.
    :param bigram_freq: The frequency of the ngram in each
        corpus as a (corpus, frequency) tuple.
    :param token_freq: The token frequency of the ngram,
        used to transform frequencies into proportions.
    :returns: Dispersion measure as a scalar.
    """
    bigram_props = [(corpus, freq / token_freq) for corpus, freq in bigram_freq]
    bigram_props = pd.DataFrame(bigram_props, columns=['corpus', 'ngram_prop'])
    bigram_props = pd.merge(
        corpus_proportions,
        bigram_props,
        on='corpus',
        how='left'
        ).fillna(0)
    kld_props = compute_functions.get_kld(bigram_props['ngram_prop'].values,
        bigram_props['corpus_prop'].values)
    return kld_props

def get_entropy_dif(ngram_1_freqs, ngram_2):
    """
    Function to obtain the difference in the entropy of a 
    slot in the ngram and the entropy if the target component
    was eliminated from the distribution.
    :param ngram_1_freqs: Frequency distribution (nltk.FreqDist)
        of the successors of the target slot.
    :param ngram_2: A string specifying the occurrying component
        of the ngram to be eliminated from the frequency distribution.
    :returns: The difference, as counterfactual - actual. Scalar.
    """
    slot_dist = np.array(list(ngram_1_freqs.values()))
    entropy = compute_functions.get_entropy(slot_dist)
    freqs_cf = ngram_1_freqs.copy()
    _ = freqs_cf.pop(ngram_2)
    freqs_cf = np.array(list(freqs_cf.values()))
    entropy_cf = compute_functions.get_entropy(freqs_cf)
    h_diff = entropy_cf - entropy
    return h_diff

def get_association(comp_1, comp_2, token_freq, unigram_frequencies):
    """
    Obtains the association between the components of a bigram as
    the KLD between joint occurrence and overall occurrence.
    Calculated both forward and backwards in the ngram.
    :param comp_1: String with the first component of the ngram.
    :param comp_2: String with the second component of the ngram.
    :param token_freq: Token frequency of the ngram.
    :unigram_frequencies: Overall unigram frequencies, summed 
        over the whole corpus. In the form of an nltk.FreqDist.
    :return: A tuple, (association_forward, association_backward).
    """
    # unigram_frequencies = processing_corpus.UNIGRAM_TOTAL
    # joint probability is conditioned on the unigram frequencies
    prob_1_2 = token_freq / unigram_frequencies[comp_1]
    prob_2_1 = token_freq / unigram_frequencies[comp_2]
    prob_1 = unigram_frequencies.freq(comp_1)
    prob_2 = unigram_frequencies.freq(comp_2)
    assoc_f = compute_functions.get_kld(np.array([prob_1_2, 1 - prob_1_2]),
        np.array([prob_2, 1 - prob_2]))
    assoc_b = compute_functions.get_kld(np.array([prob_2_1, 1 - prob_2_1]),
        np.array([prob_1, 1 - prob_1]))
    return assoc_f, assoc_b


def get_bigram_scores(ngram, forward_dict, backward_dict, unigram_dict, corpus_proportions, verbose=False):
    """
    Function for computing the MWU measures for a target ngram. 
    :param ngram: A string with the ngram to be analyzed.
    :returns: A dictionary with all MWU measures obtained:
        Token frequency, dispersion, type frequency for each slot,
        entropy difference for each slot, both directions of 
        association.
    """
    comps = ngram.split(' ')
    comp_1 = comps[0]
    comp_2 = comps[1]
    # forward_dict = processing_corpus.BIGRAM_FW
    # backward_dict = processing_corpus.BIGRAM_BW
    bigram_freq = [(corpus, corpus_dict.get(comp_1, pd.Series(0)).get(comp_2, 0)) for
        corpus, corpus_dict in forward_dict.items()]
    # Token frequency
    token_freq = sum(freqs[1] for freqs in bigram_freq)
    if token_freq == 0:
        if verbose:
            print(f'{" ".join([comp_1, comp_2])}>> is not in the corpus')
        return None
    # Dispersion
    dispersion = get_dispersion(bigram_freq, token_freq, corpus_proportions)

    ## Need aggregate frequencies for the rest
    freqs_ngram_1 = [FreqDist(dict(corpus_dict.get(comp_1, pd.Series(0)))) for corpus_dict in forward_dict.values()]
    freqs_ngram_2 = [FreqDist(dict(corpus_dict.get(comp_2, pd.Series(0)))) for corpus_dict in backward_dict.values()]
    freqs_ngram_1 = sum(freqs_ngram_1, FreqDist())
    freqs_ngram_2 = sum(freqs_ngram_2, FreqDist())

    # Type frequencies
    typef_1 = freqs_ngram_2.B()
    typef_2 = freqs_ngram_1.B()

    # Entropy
    slot1_diff = get_entropy_dif(freqs_ngram_2, comp_1)
    slot2_diff = get_entropy_dif(freqs_ngram_1, comp_2)

    # Association
    assoc_f, assoc_b = get_association(comp_1, comp_2, token_freq, unigram_dict)

    return {
        'ngram': (comp_1, comp_2), 
        'first': comp_1,
        'second': comp_2,
        'token_freq': token_freq,
        'dispersion': dispersion,
        'type_1': typef_1,
        'type_2': typef_2,
        'entropy_1': slot1_diff,
        'entropy_2': slot2_diff,
        'assoc_f': assoc_f,
        'assoc_b': assoc_b
        }


def consolidate_scores():
    """
    Function to consolidate the per-corpus bigram frequencies
    into a single distribution. Used for normalization.
    """
    global CONSOLIDATED_FW, CONSOLIDATED_BW
    CONSOLIDATED_FW = defaultdict(FreqDist)
    for _, corpus_dict in processing_corpus.BIGRAM_FW.items():
        for bigram, freq in corpus_dict.items():
            CONSOLIDATED_FW[bigram].update(freq)

    CONSOLIDATED_BW = defaultdict(FreqDist)
    for _, corpus_dict in processing_corpus.BIGRAM_BW.items():
        for bigram, freq in corpus_dict.items():
            CONSOLIDATED_BW[bigram].update(freq)


def normalize_scores(bigram_scores, entropy_limits=None, scale_entropy=False):
    """
    Normalizes the scores obtained by get_bigram_scores using the transformations
    suggested by S. Gries. 
    :param bigram_scores: A Dataframe of the type returned by get_mwu_scores.
    :param entropy_limits: Provide as a tuple or list a lower and upper
        limits for the difference in entropy measure. Useful when
        looking at a wide range of MWU, as this measure has high and low outliers that
        bias the min-max normalization. Mostly rule of thumb, but recommended values
        can be [-0.05, 0.05] or [-0.1, 0.1].
    :param scale_entropy: If True, differences in entropy are scaled with a cubic
        function to expand the range of values.
    :returns: A copy of the input DataFrame with the values normalized.
    """
    # Token: min_max
    global CONSOLIDATED_FW, CONSOLIDATED_BW

    if not CONSOLIDATED_FW and not CONSOLIDATED_BW:
        print('First time normalizing. Need to consolidate...')
        consolidate_scores()
    max_token = np.log(max([freq[freq.max()] for freq in CONSOLIDATED_FW.values() if freq.B() > 0]))
    min_token = np.log(1)

    normalized_scores = bigram_scores.copy()
    normalized_scores['token_freq'] = min_max_norm(
        np.log(bigram_scores['token_freq']),
        min_token,
        max_token
        )

    # Dispersion: substract from 1
    normalized_scores['dispersion'] = 1 - normalized_scores['dispersion']

    # Types: min_max and substract from 1
    typef_1_all = [freq.B() for freq in CONSOLIDATED_BW.values()]
    typef_1_max = np.log(max(typef_1_all))
    typef_1_min = np.log(1)

    typef_2_all = [freq.B() for freq in CONSOLIDATED_FW.values()]
    typef_2_max = np.log(max(typef_2_all))
    typef_2_min = np.log(1)

    normalized_scores['type_1'] = 1 - min_max_norm(
        np.log(bigram_scores['type_1']),
        typef_1_min,
        typef_1_max
        )
    normalized_scores['type_2'] = 1 - min_max_norm(
        np.log(bigram_scores['type_2']),
        typef_2_min,
        typef_2_max)

    # Entropy, min-max normalize. Optional: limit the range and spread values with cubic root.
    entropy_1 = bigram_scores['entropy_1']
    entropy_2 = bigram_scores['entropy_2']

    entropy_1 = entropy_1.apply(
        compute_functions.threshold_value, 
        args=(entropy_limits[0], entropy_limits[1])
        )
    entropy_2 = entropy_2.apply(
        compute_functions.threshold_value, 
        args=(entropy_limits[0], entropy_limits[1])
        )

    if scale_entropy:
        entropy_limits[0] = np.cbrt(entropy_limits[0])
        entropy_limits[1] = np.cbrt(entropy_limits[1])
        entropy_1 = np.cbrt(entropy_1)
        entropy_2 = np.cbrt(entropy_2)

    if entropy_limits:
        entropy_1 = min_max_norm(entropy_1, entropy_limits[0], entropy_limits[1])
        entropy_2 = min_max_norm(entropy_2, entropy_limits[0], entropy_limits[1])
    else:
        entropy_1 = min_max_norm(entropy_1, -1, 1)
        entropy_2 = min_max_norm(entropy_2, -1, 1)

    normalized_scores['entropy_1'] = entropy_1
    normalized_scores['entropy_2'] = entropy_2

    return normalized_scores

def corpus_to_series(corpus_dict):
    corpus_dict = {corpus: dict(ngram_dist) for corpus, ngram_dist in corpus_dict.items()} 
    corpus_series = {corpus: pd.Series({ngram: pd.Series(freqs) for ngram, freqs in corpus_freqs.items()}) for corpus, corpus_freqs in corpus_dict.items()}
    return corpus_series

def get_mwu_scores(ngrams, parallel=False, ncores=1, normalize=False, entropy_limits=None, scale_entropy=None, verbose=False, track_progress=False):
    """
    Main function to compute MWU scores for the given ngrams. Normalization is optional.
    :param ngrams: Iterable of ngrams in string form. Ngrams components should be separated by 
        a space, in lowercase and with no standalone punctuation, congruent with preprocess_corpus.
    :param normalize: Whether to normalize the MWU scores based on the complete corpus or not. 
        Because normalization requires access to the frequencies of all ngrams, the first time 
        it's run it takes longer. Subsequent uses don't have to recompute them.
    :param entropy_limits: Tuple or list containing the lower and upper boundaries of the entropy 
        difference normalization. Useful because of the skewed distribution of entropy values across 
        a corpus.
    :param scale_entropy: Whether to scale entropy differences using a cubic function to 
        expand variance. Less useful if limites are not provided, but can still clean 
        the distribution a bit.
    :returns: a dataframe with the MWU scores for each ngram provided. If normalization is true,
        a dictionary with both raw and normalized scores.
    """
    global forward_dict, backward_dict, unigram_dict, corpus_proportions
    if track_progress:
        i = 0
        # if __name__ == 'mwu_measures.mwu_functions':
        #     print('hello')
    # result = Parallel(n_jobs=4)(delayed(get_bigram_scores)(ngram) for ngram in ngrams)
    # print(result)
    
    if parallel:
        print('parallel')
        print('dumping objects...')
        forward_dict = corpus_to_series(processing_corpus.BIGRAM_FW)
        backward_dict = corpus_to_series(processing_corpus.BIGRAM_BW)
        unigram_dict = {corpus: pd.Series(frequencies) for corpus, frequencies in processing_corpus.UNIGRAM_FREQUENCIES_PC.items()}
        corpus_proportions = processing_corpus.CORPUS_PROPORTIONS

        dump(forward_dict, './memmap_cache/fwdict')
        forward_dict = load('./memmap_cache/fwdict', mmap_mode='r')
        dump(backward_dict, './memmap_cache/bwdict')
        backward_dict = load('./memmap_cache/bwdict', mmap_mode='r')
        dump(unigram_dict, './memmap_cache/unigdict')
        unigram_dict = load('./memmap_cache/unigdict', mmap_mode='r')
        dump(corpus_proportions, './memmap_cache/cproportions')
        corpus_proportions = load('./memmap_cache/cproportions', mmap_mode='r')
        # se demora mil años
        print('running multiprocess')
        partial_function = partial(get_bigram_scores, forward_dict=forward_dict, backward_dict=backward_dict,unigram_dict=unigram_dict,corpus_proportions=corpus_proportions)
        par_bigrams = lambda x: list(map(partial_function, x))
        # might be optimizable by getting input structures into numpy arrays
        bigram_chunks = [list(chunks) for chunks in np.array_split(ngrams, 4)]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            all_scores = Parallel(n_jobs=4, verbose=40, pre_dispatch='all')(delayed(par_bigrams)(chunk) for chunk in bigram_chunks)
        all_scores = [ngram for chunk in all_scores for ngram in chunk]
        all_scores = [score for score in all_scores if score] # gets rid of None
        print(all_scores)
        # if __name__ == 'mwu_measures.mwu_functions':
        #     forward_dict = processing_corpus.BIGRAM_FW
        #     backward_dict = processing_corpus.BIGRAM_BW
        #     unigram_dict = processing_corpus.UNIGRAM_TOTAL
        #     corpus_proportions = processing_corpus.CORPUS_PROPORTIONS
        #     partial_function = partial(get_bigram_scores, forward_dict=forward_dict, backward_dict=backward_dict,unigram_dict=unigram_dict,corpus_proportions=corpus_proportions)
        #     par_bigrams = lambda x: list(map(partial_function, x))
        #     bigram_chunks = [list(chunks) for chunks in np.array_split(ngrams, 4)]
        #     print(bigram_chunks)
        #     with Pool(4) as pool:
        #         # args = zip(ngrams, repeat(forward_dict), repeat(backward_dict), repeat(unigram_dict), repeat(corpus_proportions))
        #         all_scores = pool.map(par_bigrams, bigram_chunks)
        #     all_scores = list(all_scores)
        #     all_scores = [ngram for chunk in all_scores for ngram in chunk]
        #     all_scores = [score for score in all_scores if score] # gets rid of None
        #     c = Client()
        #     webbrowser.open(c.dashboard_link)
        #     input_ngrams = db.from_sequence(ngrams, npartitions=npartitions)
        #     all_scores = input_ngrams.map(lambda x: get_bigram_scores(
        #         x,
        #         forward_dict,
        #         backward_dict,
        #         unigram_dict,
        #         corpus_proportions,
        #         verbose
        #     ))
        #     c.shutdown()

    else:
        forward_dict = processing_corpus.BIGRAM_FW
        backward_dict = processing_corpus.BIGRAM_BW
        unigram_dict = processing_corpus.UNIGRAM_TOTAL
        corpus_proportions = processing_corpus.CORPUS_PROPORTIONS

        all_scores = []
        for ngram in ngrams:
            if verbose:
                print(ngram)
            if track_progress:
                i += 1
                if i % 1000 == 0:
                    print(f'{i} ngrams processed')

            #TODO: Trigram info
            bigram_scores = get_bigram_scores(
                ngram,
                forward_dict,
                backward_dict,
                unigram_dict,
                corpus_proportions,
                verbose
                )
            if bigram_scores:
                all_scores.append(bigram_scores)

    results_dataframe = pd.DataFrame(all_scores)
    results_dataframe['ngram'] = results_dataframe['ngram'].apply(' '.join)

    if normalize:
        results_norm = normalize_scores(results_dataframe, entropy_limits, scale_entropy)
        return {
            'raw': results_dataframe, 
            'normalized': results_norm
            }

    return results_dataframe
