import pandas as pd
import numpy as np
from .book import BookText

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer


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
        wc = bt.word_count(rem_stopwords=False)
        wc_nostop = bt.word_count(rem_stopwords=True)
        sent = bt.sentence_count()
        return pd.Series([wc / sent, wc_nostop / sent],
                         index=['word_per_sent', 'word_per_sent_nostop'])


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


class POSFeatures(BaseEstimator, TransformerMixin):
    """Converts text to POS features"""

    def __init__(self):
        pass

    def transform(self, X):
        # do whatever is needed to build features
        return transformed_X

    def fit(self, X, y=None):
        return self


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