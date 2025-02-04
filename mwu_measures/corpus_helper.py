import pandas as pd
default_weights = {'token_freq': 1/8, 'dispersion': 1/8, 'type_1': 1/8, 'type_2': 1/8, 'entropy_1': 1/8, 'entropy_2': 1/8, 'fw_assoc': 1/8, 'bw_assoc': 1/8}


class Fetcher():
    def __init__(self, corpus):
        self.corpus = corpus
        
    def __call__(self, query, df = True):
        if df:
            return(self.corpus.df(query))
        else:
            return(self.corpus(query))

    def query(self, query):
        self.corpus(query)

    def query_corpus(self, ngrams, source, target):
        self.corpus.create_query(ngrams, source, target)

    def create_scores(self, ngrams, from_text=False):
        ngrams = list(set(ngrams))
        bigrams = []
        trigrams = []

        for ngram in ngrams:
            split_ngram = ngram.split()
            if len(split_ngram) == 2:
                bigrams.append(tuple(split_ngram))
            elif len(split_ngram) == 3:
                if not from_text:
                    bigrams.append((split_ngram[0], split_ngram[1]))
                trigrams.append((' '.join((split_ngram[0], split_ngram[1])), split_ngram[2]))
            else:
                print(split_ngram)
        bigrams = [list(bigram) for bigram in set(bigrams)]
        trigrams = [list(trigram) for trigram in set(trigrams)]
        self.query_corpus(bigrams, 'ug_1', 'ug_2')
        self.bigram_scores = self.corpus.get_ngram_scores('ug_1', 'ug_2', 2)
        self.query_corpus(trigrams, 'big_1', 'ug_3')
        self.trigram_scores = self.corpus.get_ngram_scores('big_1', 'ug_3', 3)

    def get_measures(self, ngram, normalized=True):
        if normalized:
            mode = 'normalized'
        else:
            mode = 'raw'
        split_ngram = ngram.split()
        ug_1 = split_ngram[0]
        ug_2 = split_ngram[1]
        if len(split_ngram) == 2:
            mwu_measures = self.bigram_scores[mode].loc[(self.bigram_scores[mode].ug_1 == ug_1) & (self.bigram_scores[mode].ug_2 == ug_2)]
            mwu_measures = mwu_measures.rename(columns={'ug_1': 'comp_1', 'ug_2': 'comp_2'})
            if len(mwu_measures) == 0:
                print(f"No data for bigram {ngram}")
                return None
            else: 
                return mwu_measures
        elif len(split_ngram) == 3:
            big_1 = ' '.join([ug_1, ug_2])
            ug_3 = split_ngram[2]
            first_mwu = self.bigram_scores[mode].loc[(self.bigram_scores[mode].ug_1 == ug_1) & (self.bigram_scores[mode].ug_2 == ug_2)]
            first_mwu = first_mwu.rename(columns={'ug_1': 'comp_1', 'ug_2': 'comp_2'})
            second_mwu = self.trigram_scores[mode].loc[(self.trigram_scores[mode].big_1 == big_1) & (self.trigram_scores[mode].ug_3 == ug_3)]
            second_mwu = second_mwu.rename(columns={'big_1': 'comp_1', 'ug_3': 'comp_2'})
            if len(first_mwu) == 0 or len(second_mwu) == 0:
                print(f"No data for trigram {ngram}")
                return None
            else:
                return pd.concat([first_mwu, second_mwu], axis=0).reset_index(drop=True)
    
    def weight_measures(self, mwu_measures, weight_dict=default_weights):
        for col in mwu_measures.columns:
            if col in weight_dict.keys():
                mwu_measures[col] = mwu_measures[col] * weight_dict[col]
        return(mwu_measures)

    def get_score(self, ngram, weights=default_weights):
        this_measure = self.get_measures(ngram, normalized=True)
        if this_measure is not None:
            weighted = self.weight_measures(this_measure, weights)
            mwu_score = weighted.drop('ngram_length', axis=1).sum(axis=1, numeric_only=True)
            if len(mwu_score) == 1:
                return mwu_score.iloc[0]
            else:
                return tuple(mwu_score)
        else:
            return None

    def get_measures_batch(self, ngrams, normalized=True, from_text=False):
        bigrams = []
        trigrams = []
        for ngram in ngrams:
            split_ngram = ngram.split()
            if len(split_ngram) == 2:
                bigrams.append((split_ngram[0], split_ngram[1]))
            elif len(split_ngram) == 3:
                if not from_text:
                    bigrams.append((split_ngram[0], split_ngram[1]))
                trigrams.append((' '.join((split_ngram[0], split_ngram[1])), split_ngram[2]))
        bigrams = pd.DataFrame(bigrams, columns=['ug_1', 'ug_2'])
        trigrams = pd.DataFrame(trigrams, columns=['big_1', 'ug_3'])
        if normalized:
            mode = 'normalized'
        else:
            mode = 'raw'
        bigram_scores = pd.merge(self.bigram_scores[mode], bigrams, how='inner')
        trigram_scores = pd.merge(self.trigram_scores[mode], trigrams, how='inner')
        bigram_scores = bigram_scores.rename(columns={'ug_1': 'comp_1', 'ug_2': 'comp_2'})
        trigram_scores = trigram_scores.rename(columns={'big_1': 'comp_1', 'ug_3' : 'comp_2'})
        return pd.concat([bigram_scores, trigram_scores], axis=0).reset_index(drop=True)
    def get_score_batch(self, ngrams, weights = default_weights, from_text=False, normalized=True):
        self.create_scores(ngrams)
        this_measure = self.get_measures_batch(ngrams, normalized=normalized, from_text=from_text)
        if not normalized:
            return this_measure
        else:
            weighted = self.weight_measures(this_measure, weights)
            mwu_score = weighted.drop('ngram_length', axis=1).sum(axis=1, numeric_only=True)
            weighted['mwu_score'] = mwu_score
            return (weighted[['ngram', 'comp_1', 'comp_2', 'mwu_score', 'ngram_length']], this_measure)