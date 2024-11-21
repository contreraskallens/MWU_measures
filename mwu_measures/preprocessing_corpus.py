"""
Module for transforming a corpus into the format expected by the process_corpus module.
The functions should return a dictionary of the different corpora to be included.
Dictionaries are of the type {Corpus: nltk.FreqDist}.
Here I provided procedures to preprocess a couple of common corpora.
Anything other than that should be included by user.
"""

import re
from collections import defaultdict, Counter
from itertools import groupby
from nltk import flatten
from nltk.util import trigrams as get_trigrams


def clean_bnc_line(this_line):
    """
    Takes a line from the tokenized BNC corpus and returns a list of cleaned tokens.
    """
    this_line = re.sub(r'^.+\t', '', this_line).lower()
    this_line = re.sub(r" (n't|'s|'ll|'d|'re|'ve|'m)", r'\1', this_line)
    this_line = this_line.replace('wan na', 'wanna')
    this_line = this_line.replace('\n', '')
    this_line = this_line.replace('-', '')
    # Get rid of standalone punctuation and double (and more) spaces
    this_line = re.sub(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', this_line).strip()
    this_line = re.sub(r'\s\W+\s|\s\W+|^\W\s$|\s+', ' ', this_line).strip()
    this_line = this_line.split()
    return this_line

def make_bigram_dict():
    this_dict = defaultdict(Counter)
    return this_dict

def preprocess_test():
    # unigram_freqs = defaultdict(Counter)
    # trigram_counts = defaultdict(Counter)
    # trigram_fw =  defaultdict(lambda: defaultdict(make_bigram_dict))
    # trigram_bw = defaultdict(lambda: defaultdict(make_bigram_dict))
    # n_trigrams = 0
    with open('mwu_measures/corpora/test_corpus.txt', 'r', encoding="utf-8") as corpus_file:
        raw_lines = corpus_file.read().splitlines()
    # unigrams = {key:[clean_bnc_line(line) for line in group] for key, group in org_items
    #             }
    # trigrams = {
    #     key:[trigram for line in group for trigram in get_trigrams(line) if len(line) > 1]
    #     for key, group in unigrams.items()
    # }
    # trigrams = {corpus: Counter(trigrams) for corpus, trigrams in trigrams.items()}
    # trigrams = [(corpus, ngram[0], ngram[1], ngram[2], freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]

    # unigrams = {corpus: Counter(flatten(unigrams)) for corpus, unigrams in unigrams.items()}
    # unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]
    # return unigrams, trigrams

    split_lines = [line.split() for line in raw_lines]
    # unigrams = [unigram for unigram in ]
    # unigrams = zip(['A', 'B', 'C'], split_lines)
    trigrams = [Counter(get_trigrams(line)) for line in split_lines]
    corpora = ['A', 'B', 'C']
    trigrams = [(corpus, trigram[0], trigram[1], trigram[2], freq) for corpus, corpus_dict in zip(corpora, trigrams) for trigram, freq in corpus_dict.items()]
    unigrams = [Counter(unigrams) for unigrams in split_lines]
    unigrams = [(corpus, unigram, freq) for corpus, corpus_dict in zip(corpora, unigrams) for unigram, freq in corpus_dict.items()]
    return unigrams, trigrams


# def preprocess_bnc(bnc_dir, chunk_size = 10000, verbose = False):
def preprocess_bnc(raw_lines):
    """
    Processes the bnc_tokenized.txt file from the BNC corpus 
    to prepare for extracting distributions.
    Not normally used by itself but called from process_corpus.
    :param bnc_dir: The directory of the bnc_tokenized.txt file.
    :param chunk_size: The size (in bytes) of the file to be preprocessed at once.
        Larger numbers use more memory but are faster unless they fill RAM.
    :param verbose: Whether to print progress reports on preprocessing.
    :returns: Tuple with dictionaries for processing. First is the unigrams,
        second is the bigrams.
        The format of each dictionary is {Corpus: nltk.FreqDist} with frequency
        of each element within that corpus.
    """
    # #for BNC, you want to provide the bnc_tokenized.txt file
    org_items = groupby(raw_lines, key=lambda x: re.match(r'(^.)', x).group(1))
    unigrams = {key:[clean_bnc_line(line) for line in group] for key, group in org_items
                }
    trigrams = {
        key:[trigram for line in group for trigram in get_trigrams(line) if len(line) > 1]
        for key, group in unigrams.items()
    }
    trigrams = {corpus: Counter(trigrams) for corpus, trigrams in trigrams.items()}
    trigrams = [(corpus, ngram[0], ngram[1], ngram[2], freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]

    unigrams = {corpus: Counter(flatten(unigrams)) for corpus, unigrams in unigrams.items()}
    unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]
    return unigrams, trigrams
