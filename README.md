# Multiword unit measures
Package of functions to obtain MWU-scores for chunks.
Based on the proposal by Stefan Gries, "Multi-word units (and tokenization more generally): a multi-dimensional and largely information-theoretic approach", https://journals.openedition.org/lexis/6231.

# How to use this package

The package is based around two main functions: `process_corpus` `and get_mwu_scores`.
* `process_corpus` takes a corpus divided in lines, tokenizes it, and takes frequency distributions for (for now) unigrams and bigrams.
* `get_mwu_scores` takes an iterable (e.g. list) of ngrams (for now: just bigrams) and returns a dataframe with all the measures specified by Gries. Normalization is optional but recommended.

You can see how to use this script in `example.ipynb`.