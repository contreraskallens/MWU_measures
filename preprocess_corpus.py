import re
from collections import defaultdict
from nltk import FreqDist, flatten
from nltk.util import bigrams as get_bigrams
from nltk.util import trigrams as get_trigrams
from itertools import groupby

# Objective for preprocessing is to turn a corpus into a list of sentences/paragraphs/etc divided by corpus section. Here I provided procedures to preprocess a couple of common corpora.

def clean_bnc_line(this_line):
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

def preprocess_bnc(bnc_dir, chunk_size = 10000, verbose = False):
    #for BNC, you want to provide the bnc_tokenized.txt file
    print('Reading and cleaning corpus...')
    unigram_freqs = defaultdict(FreqDist)
    bigram_freqs = defaultdict(FreqDist)
    with open('bnc_tokenized.txt', 'r') as corpus_file:
        i = 0
        while True:
            raw_lines = corpus_file.readlines(chunk_size)
            if not raw_lines:
                break
            org_items = groupby(raw_lines, key=lambda x: re.match(r'(^.)', x).group(1))
            unigrams = {key: [clean_bnc_line(line) for line in group] for key, group in org_items}
            bigrams = {key: [bigram for line in group for bigram in get_bigrams(line) if len(line) > 1] for key, group in unigrams.items()}
            unigrams = {key: flatten(group) for key, group in unigrams.items()}
            for corpus, unigrams in unigrams.items():
                this_dist = FreqDist(unigrams)
                unigram_freqs[corpus].update(this_dist)
            for corpus, bigrams in bigrams.items():
                this_dist = FreqDist(bigrams)
                bigram_freqs[corpus].update(this_dist)
            
            if verbose:
                i += len(raw_lines)
                print(f'{i} lines processed')

    return (unigram_freqs, bigram_freqs)