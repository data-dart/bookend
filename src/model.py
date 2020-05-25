import pandas as pd
import numpy as np
from book import BookText
import textstat
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import collections

def _convert_to_dataframe(text):
    """Converts text to a DataFrame as a preprocessing step

    Accepts either a single string, a list of strings, or a
    DataFrame or Series that includes one column of strings
    named 'text'

    Useful as a first step in fit and transform methods
    """
    # if you passed a single string
    if isinstance(text, str):
        text = [text]
    # making sure it's a DataFrame
    text_frame = pd.DataFrame(text)
    # if only one column, assumes it's a text column
    if text_frame.shape[1] == 1:
        text_frame.columns = ['text']
    if 'text' not in text_frame.columns:
        raise ValueError('If passing a DataFrame, make sure'
                         'one of the columns is named "text')
    return text_frame


class LexicalFeatures(BaseEstimator, TransformerMixin):
    """Converts text into lexical features

    Returns a DataFrame with one feature vector per text
    """

    def __init__(self, include_all_features=True):
        """Contains an example hyperparameter"""
        self.include_all_features = include_all_features

    def transform(self, X):
        """Transforms the text into features"""
        text_frame = _convert_to_dataframe(X)
        return text_frame.apply(self._lexical_features_one_row, axis=1)

    def fit(self, X, y):
        return self

    def _lexical_features_one_row(self, row):
        """Creates a feature vector from one text sample"""
        bt = BookText(rawtext=row['text'])
        bt.clean(lemmatize=False, inplace=True)

        # sentences
        sens = bt.tokenize('sent', rem_stopwords=False, include_punctuation=False)
        # words
        all_words = bt.tokenize('word', rem_stopwords=False)
        # words (no stopwords)
        all_words_nostop = bt.tokenize('word', rem_stopwords=True)
        # unique lemmatizations, no stopwords
        all_words_lemmatized_nostop = bt.clean(lemmatize=True).tokenize('word', 
                                                                        rem_stopwords=True, 
                                                                        include_punctuation=False)
        unique_word_frac = len(np.unique(all_words_lemmatized_nostop)) / len(all_words_nostop)
        # for use in textstat. Removing trailing whitespace 
        # improves textstat's results
        reco = '. '.join(sens).strip()

        fk_score = textstat.flesch_kincaid_grade(reco)
        word_count = len(all_words)
        word_count_nostop = len(all_words_nostop)
        sen_count = len(sens)

        words_per_sentence_nostop = word_count_nostop / sen_count

        syllable_count_nostop = textstat.syllable_count(' '.join(all_words_nostop))

        syllables_per_word_nostop = syllable_count_nostop / word_count

        frac_stop = (word_count - word_count_nostop) / word_count

        word_lens = [len(w) for w in all_words]
        mean_word_length = np.mean(word_lens)
        word_lens_nostop = [len(w) for w in all_words_nostop]
        mean_word_length_nostop = np.mean(word_lens_nostop)

        # measures of dispersion in word lengths
        # ddof = 1.5 is a better correction, see
        # https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Rule_of_thumb_for_the_normal_distribution
        word_length_spread = np.std(word_lens, ddof=1.5)
        word_length_spread_nostop = np.std(word_lens_nostop, ddof=1.5)

        # skew in the distribution of word lengths
        word_length_skew_nostop = skew(word_lens_nostop, bias=False)

        sent_lens = [len(s.split()) for s in sens]
        sent_length_spread = np.std(sent_lens, ddof=1.5)

        return pd.Series([frac_stop, words_per_sentence_nostop, 
                              syllables_per_word_nostop,
                              mean_word_length, mean_word_length_nostop,
                              word_length_spread, word_length_spread_nostop,
                              word_length_skew_nostop, sent_length_spread, 
                              fk_score, unique_word_frac],
                          index=['frac_stop', 'words_per_sentence_nostop', 
                                    'syllables_per_word_nostop', 'mean_word_length',
                                    'mean_word_length_nostop', 'word_length_spread', 
                                    'word_length_spread_nostop', 'word_length_skew_nostop',
                                    'sent_length_spread', 'fk_score', 'unique_words'])


class BOWFeatures(BaseEstimator, TransformerMixin):
    """
    Class for fitting a Bag-of-words model on the data
    
    Usage Example in a pipeline:
    
    bow_pipe = Pipeline([('build', BOWFeatures()),
                    ('lr_bow', LogisticRegression(max_iter = 500, C = 100))
                    ])                


    """

    def __init__(self, bow={}):
        """
        :vocabulary: is a bag of words as a dictionary if it is available.
        
        """
        self.bow = bow
        

    def transform(self, X):
        """vectorizes a row of text data using the bag of words"""
        
        X = np.array(X)
        if X.ndim > 1:
            X = X.ravel()

        vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                                ngram_range=(1, 1), vocabulary=self.bow)
        
        XX = vectorizer.fit_transform(X)
        
        return XX
        

    def fit(self, X, y=None):
        """Fit DataFrame by building bag of words"""
        text_frame = _convert_to_dataframe(X)
        if self.bow =={}:
            self.build_bow(corpus = text_frame['text'].values)
        return self

    def build_bow(self, corpus):
        """
        Returns a dict of unique words in the corpus

        Input:
        A corpus in the form of a np. array

        Output: 
        Corpus vocabulary as a dictionary.

        dict_key : word
        dict_value : an integer label for the key

        """
        #---------------STEP-1: clean, tokenize, lower-------------------------------------------
        words = []
        for i in range(len(corpus)):
            b = BookText(rawtext = corpus[i])
            bb = b.clean(deromanize=True, lemmatize=True) 
            words += bb.tokenize(on='words', rem_stopwords=True)
        words = [w.lower() for w in words]

        words = list(set(words)) #Making a set early on reduces the size to speed things up


        #---------------STEP-2: remove all numeric characters ------------------------------------
        no_nums = []
        for word in words:
            if (not any(ch.isdigit() for ch in word)):
                no_nums.append(word)

        #---------------STEP-3: Package as a dictionary ------------------------------------------
        bow = no_nums
        vocab_dict = {}
        for i in range(len(bow)):
            vocab_dict[bow[i]] = i
        
        self.bow = vocab_dict
        
        
class SyntacticFeatures(BaseEstimator, TransformerMixin):
    
    """ Defining my own POS tags. """
    nouns = ['NN', 'NNS', 'NNP', 'NNPS']   
    adjectives = ['JJ', 'JJR', 'JJS']
    verbs = ['MD', 'VB', 'VBG', 'VBN', 'VBP', 'VBZ']
    adverbs = ['RB', 'RBR', 'RBS', 'WRB']
    pronouns = ['PRP', 'PRP$', 'WP', 'WP$']
    determiners = ['DT', 'PDT', 'WDT']
    conjunctions = ['CC', 'IN']
    
    """ Arrays to store different features"""
    global pos_arr, columns, pos_bigram_ordered_pairs, pos_trigram_ordered_pairs, final_columns
    
    pos_arr = np.array([['NOUN', nouns], ['ADJ', adjectives], ['VERB', verbs], ['ADV', adverbs], 
                        ['PRN', pronouns], ['DET', determiners], ['CONJ', conjunctions]])
    
    columns=[pos_arr[i][0] for i in range(np.size(pos_arr,0))]
    pos_bigram_ordered_pairs = [(columns[i], columns[j]) for i in range(len(columns)) for j in range(len(columns))]
    pos_trigram_ordered_pairs = [(columns[i], columns[j], columns[k]) 
                                 for i in range(len(columns)) for j in range(len(columns)) 
                                 for k in range(len(columns))]

    """ Final set of features which will be returned upon call. """
    
    final_columns = columns + pos_bigram_ordered_pairs + pos_trigram_ordered_pairs
    
    def __init__(self):
        test_var = 10
        
    def transform(self, X):
        """Transforms the text into syntactic features"""
        text_frame = _convert_to_dataframe(X)
        return text_frame.apply(self._syntactic_features_one_row, axis=1)

    def fit(self, X, y):
        return self
    
    def _syntactic_features_one_row(self, row):
        """Creates a feature vector for one text sample"""
        bt = BookText(rawtext=row['text'])
        bt = bt.clean(lemmatize=False, inplace=False)
        
        """ This gives the distribution for n-grams generated from POS."""

        def get_pos_ngrams(bt: BookText):
            pos = bt.translate_to_pos()
            pos_as_arr = pos.tokenize('word',rem_stopwords=False, include_punctuation=False)
            pos_unigram = ngrams(pos_as_arr, 1)
            pos_bigrams = ngrams(pos_as_arr, 2)
            pos_trigrams = ngrams(pos_as_arr, 3)
            pos_unigram_freq = collections.Counter(pos_unigram)
            pos_bigram_freq = collections.Counter(pos_bigrams)
            pos_trigram_freq = collections.Counter(pos_trigrams)
            pos_unigram_counter = np.zeros(len(columns))
            pos_bigram_counter = np.zeros(len(pos_bigram_ordered_pairs))
            pos_trigram_counter = np.zeros(len(pos_trigram_ordered_pairs))
            
            for i, elem in enumerate(columns):
                pos_unigram_counter[i] = pos_unigram_freq[elem]

            for i, elem in enumerate(pos_bigram_ordered_pairs):
                pos_bigram_counter[i] = pos_bigram_freq[elem]

            for i, elem in enumerate(pos_trigram_ordered_pairs):
                pos_trigram_counter[i] = pos_trigram_freq[elem]

            pos_ngram_counter = np.append(pos_unigram_counter, pos_bigram_counter)
            pos_ngram_counter = np.append(pos_ngram_counter, pos_trigram_counter)
            return pos_ngram_counter
        
        n_gram_counts = get_pos_ngrams(bt)
        word_count = bt.word_count(rem_stopwords = False, include_punctuation=True)
        
        if word_count != 0:
            n_gram_counts = n_gram_counts/word_count

        return pd.Series(n_gram_counts, index=final_columns)
        
class NGramFeatures(BaseEstimator, TransformerMixin):
    """Converts text into n-gram features"""

    def __init__(self):
        pass

    def transform(self, X):
        # does whatever is needed to build features
        # presumably, uses self.topic_graph
        return transformed_X

    def fit(self, X, y=None):
        # builds the n-gram graph or graphs
        text_frame = _convert_to_dataframe(X)
        self.build_graph(corpus=' '.join(text_frame['text']))
        return self

    def build_graph(self, corpus):
        self.topic_graph = derived_graph



# an example usage incorporating these into a pipe
# lexical_pipe = Pipeline([('features', Pipeline([
#                               ('build', LexicalFeatures()),
#                               ('scale', StandardScaler())
#                               ])
#                           ),
#                          ('svc', SVC())])


# eventually, we'll build a bunch of pipes,
# e.g. lexical_pipe, bow_pipe, pos_pipe, ngram_pipe
# if we want to exclude any of these from the model, we can set
# e.g. model.set_params(lexical='drop'), and it will drop them
# model = VotingClassifier([('lexical', lexical_pipe),
#                           ('bow', bow_pipe)], 
#                          voting='soft')