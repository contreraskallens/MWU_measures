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

def preprocess_bnc(bnc_dir, chunk_size = 10000, verbose = False):
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
    #for BNC, you want to provide the bnc_tokenized.txt file
    if verbose:
        print('Reading and cleaning corpus...')
    # Corpus, uni_1, uni_2, uni_3, frequency
    unigram_freqs = defaultdict(Counter)
    trigram_fw =  defaultdict(lambda: defaultdict(make_bigram_dict))
    trigram_bw = defaultdict(lambda: defaultdict(make_bigram_dict))
    with open(bnc_dir, 'r', encoding="utf-8") as corpus_file:
        i = 0
        while True:
            raw_lines = corpus_file.readlines(10000)
            if not raw_lines:
                break
            org_items = groupby(raw_lines, key=lambda x: re.match(r'(^.)', x).group(1))
            unigrams = {
                key:[clean_bnc_line(line) for line in group] for key, group in org_items
                }
            for corpus, unigram_list in unigrams.items():
                chunk_unigrams = flatten(unigram_list)
                unigram_freqs[corpus].update(chunk_unigrams)
            trigrams = {
                key:[bigram for line in group for bigram in get_trigrams(line) if len(line) > 1]
                for key, group in unigrams.items()
                }
            trigram_counts = {corpus: Counter(trigram_ocs) for corpus, trigram_ocs in trigrams.items()} #dunno if this step is necessary
            for corpus, counts in trigram_counts.items():
                for trigram, freq in counts.items():
                    trigram_fw[corpus][trigram[0]][trigram[1]].update([trigram[2]])
                    trigram_bw[corpus][trigram[2]][trigram[1]].update([trigram[0]])
            i += 1
            if verbose:
                i += len(raw_lines)
                print(f'{i} lines processed')
    print('Merging....')        
    trigram_merged_bw = {corpus: {this_c3: Counter({(this_c2, this_c1): freq for this_c2, c1_freqs in c2_freqs.items() for this_c1, freq in c1_freqs.items()})  for this_c3, c2_freqs in c3_freqs.items()} for corpus, c3_freqs in trigram_bw.items()}
    return (unigram_freqs, trigram_fw, trigram_bw, trigram_merged_bw)
