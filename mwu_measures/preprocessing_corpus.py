"""
Module for transforming a corpus into the format expected by the process_corpus module.
The functions should return a dictionary of the different corpora to be included.
Dictionaries are of the type {Corpus: nltk.FreqDist}.
Here I provided procedures to preprocess a couple of common corpora.
Anything other than that should be included by user.
"""

from collections import defaultdict, Counter
from itertools import groupby, islice, tee
from nltk.util import trigrams as get_trigrams
from nltk import ngrams
import pandas as pd
import regex
import numpy as np
import pandas as pd

def clean_bnc_lines(raw_lines):
    """
    Takes a line from the tokenized BNC corpus and returns a list of cleaned lines
    """
    raw_lines = pd.Series(raw_lines)
    corpus_list = raw_lines.str.extract(r'(^.)', expand=False)
    processed_lines = raw_lines.str.replace(r'^.+\t', '', regex=True)
    processed_lines = processed_lines.str.lower()
    processed_lines = processed_lines.str.replace(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", regex=True)
    processed_lines = processed_lines.str.replace('wan na', 'wanna', regex = False)
    processed_lines = processed_lines.str.replace('\n', '')
    processed_lines = processed_lines.str.replace('-', '')
    processed_lines = processed_lines.str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(r'\s*\W+\s*', ' ', regex=True)
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(r'\s+', ' ', regex=True)
    processed_lines = 'START ' + processed_lines + ' END'
    return list(zip(corpus_list, processed_lines.to_list()))

def clean_coca_lines_regex(raw_lines, corpus_ids):
    processed_lines = regex.sub(r' [\.\?\!] |\n|(@ )+|</*[ph]>|<br>', ' splitmehere ', raw_lines.lower())
    processed_lines = regex.sub(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", processed_lines)
    processed_lines = regex.sub(r'@@\d+\s*', r"", processed_lines)
    processed_lines = processed_lines.replace('wan na', 'wanna')
    processed_lines = processed_lines.replace('-', ' ')
    processed_lines = regex.sub(r'\d+', ' NUMBER ', processed_lines)
    processed_lines = regex.sub(r' \W|\W ', ' ', processed_lines)
    processed_lines = regex.sub(r'\s+', ' ', processed_lines)
    processed_lines = regex.split(r'\s*splitmehere\s*', processed_lines)
    processed_lines = [line for line in processed_lines if len(line) > 0] # get rid of empty lines
    # Might get some mileage out of converting to Series, doing vectorized, and then converting back to list.
    # Get rid of double spaces and trailing spaces, add header and footer in line
    processed_lines = ['START START ' + ' '.join(line.split()).strip() + ' END END' for line in processed_lines if len(line) > 0]
    processed_lines = (corpus_ids, ' '.join(processed_lines))
    return [processed_lines]
    # return list(zip([corpus_ids] * len(processed_lines), processed_lines))

def make_bigram_dict():
    this_dict = defaultdict(Counter)
    return this_dict

def generate_trigrams(text):
    words = text.split()
    return list(zip(words, islice(words, 1, None), islice(words, 2, None)))

def pairwise(iterable, n=2):
    return zip(*(islice(it, pos, None) for pos, it in enumerate(tee(iterable, n))))

def zipngram2(text, n=2):
    words = text.split()
    return pairwise(words, n)

def preprocess_test():
    with open('mwu_measures/corpora/test_corpus.txt', 'r', encoding="utf-8") as corpus_file:
        raw_lines = corpus_file.read().splitlines()
    split_lines = [line.split() for line in raw_lines]
    trigrams = [Counter(zipngram2(line, 4)) for line in raw_lines]
    corpora = ['A', 'B', 'C']
    trigrams = [(corpus, trigram[0], trigram[1], trigram[2], trigram[3], freq) for corpus, corpus_dict in zip(corpora, trigrams) for trigram, freq in corpus_dict.items()]
    unigrams = [Counter(unigrams) for unigrams in split_lines]
    unigrams = [(corpus, unigram, freq) for corpus, corpus_dict in zip(corpora, unigrams) for unigram, freq in corpus_dict.items()]
    return unigrams, trigrams

def preprocess_corpus(raw_lines, corpus, corpus_ids=None):
    # corpus_id is for CoCA
    # for BNC, you want to provide the bnc_tokenized.txt file
    # for coca, folder with texts
    if corpus == 'bnc':
        this_lines = clean_bnc_lines(raw_lines)
    elif corpus == 'coca' or corpus == 'coca_sample':
        print('cleaning...')
        this_lines = clean_coca_lines_regex(raw_lines, corpus_ids)
    print('extracting trigrams...')
    all_trigrams = {key: zipngram2(corpus, 4) for key, corpus in this_lines}
    trigrams = {key: Counter(list(trigrams)) for key, trigrams in all_trigrams.items()}
    # Can't find anything remotely faster than this
    trigrams = [(corpus, *ngram, freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]
    unigrams = {corpus: Counter(corpus_lines.split()) for corpus, corpus_lines in this_lines}
    unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]
    return unigrams, trigrams



    # org_items = groupby(this_lines, key=lambda x: x[0])
    # print('extracting trigrams...')
    # unigrams = {key:[line[1] for line in group] for key, group in org_items}
    # # all_trigrams = {key: [' '.join(trigram) for line in group for trigram in list(generate_trigrams(line))] for key, group in unigrams.items()}
    # all_trigrams = {key: [' '.join(trigram) for line in group for trigram in zipngram2(line, 3)] for key, group in unigrams.items()}
    # trigrams = {key: Counter(trigrams) for key, trigrams in all_trigrams.items()}    
    # # trigrams =  {key: dict(zip(*np.unique(trigrams, return_counts=True))) for key, trigrams in all_trigrams.items()}
    # trigrams = [(corpus, ngram, freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]
    # unigrams = {corpus: Counter([unigram for line in corpus_lines for unigram in line.split()]) for corpus, corpus_lines in unigrams.items()}
    # unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]