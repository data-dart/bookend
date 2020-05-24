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
    """Converts text into bag of words features

    More description
    """

    def __init__(self):
        pass

    def transform(self, X):
        """Transforms data into features, assuming self.bow exists"""
        text_frame = _convert_to_dataframe(X)
        return text_frame.apply(self._compare_bow_one_row, axis=1)

    def fit(self, X, y=None):
        """Fit DataFrame by building bag of words"""
        text_frame = _convert_to_dataframe(X)
        self.build_bow(corpus=' '.join(text_frame['text']))
        return self

    def build_bow(self, corpus):
        """build a bag of words from a list of words"""
        # do whatever
        # figure out what derived_bag_of_words is
        self.bow = derived_bag_of_words

    def _compare_bow_one_row(self, text):
        """This does whatever bag of words comparison it needs"""
        pass


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
lexical_pipe = Pipeline([('features', Pipeline([
                              ('build', LexicalFeatures()),
                              ('scale', StandardScaler())
                              ])
                          ),
                         ('svc', SVC())])


# eventually, we'll build a bunch of pipes,
# e.g. lexical_pipe, bow_pipe, pos_pipe, ngram_pipe
# if we want to exclude any of these from the model, we can set
# e.g. model.set_params(lexical='drop'), and it will drop them
model = VotingClassifier([('lexical', lexical_pipe),
                          ('bow', bow_pipe)], 
                         voting='soft')