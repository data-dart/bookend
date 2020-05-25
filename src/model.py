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
import networkx as nx


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
        
# This has not been tested yet        
class NGramFeatures(BaseEstimator, TransformerMixin):
    """Converts text into n-gram features"""

    def __init__(self, ngram_instr):
        """ngram_instr is a list of tuples with instructions on how to create the ngram graphs and features
           format: [(n_1, wordgram_1, pos_1),(n_2, wordgram_2, pos_2),...]
           example: If you wanted uni-word-grams using POS tagged text you would pass ngram_instr=[(1, True, True)]
        """
        self.ngrams = ngram_instr

    def transform(self, X):
        # does whatever is needed to build features
        # presumably, uses self.topic_graph
        text_frame = _convert_to_dataframe(X)
        for i,instr in enumerate(self.ngrams):
            # Making the graphs
            self.make_graphs(text_frame, n=instr[0], wordgram=instr[1], pos=instr[2])
            # Building the column name that has the appropriate graph
            frame_key = 'graphs_'
            if instr[2]:
                new_key = 'pos_'+str(instr[0])
            else:
                new_key = str(instr[0])
            frame_key+=new_key

            self.get_graph_metrics(text_frame, self.topic_graphs[i], frame_key, new_key+'_')

        transformed_X = text_frame.copy(deep=True)
        # Removing the columns that were created in order to generate the features
        transformed_X.drop(columns=['author','text','book'])
        for column in transformed_X.columns:
            if ('graph' in column):
                transformed_X.drop(columns=column)
        return transformed_X

    def fit(self, X, y=None, random_seed=None, tdsplit=0.5):
        # builds the n-gram graph or graphs
        text_frame = _convert_to_dataframe(X)
        self.topic_graphs = np.empty(len(self.ngrams))
        for i,instr in enumerate(self.ngrams):
            topic_graphs_temp = {}
            self.make_graphs(text_frame, n=instr[0], wordgram=instr[1], pos=instr[2])
            if y is not None:
                for author in np.unique(y):
                    # If pos=True need extra string to identify graph
                    if instr[2]: 
                        pos_string = 'pos_'
                    else:
                        pos_string = ''
                    self.topic_graphs[author] = self.make_topic_graph(list(text_frame[y == author]['graphs_'+pos_string+str(instr[0])]))
            else:
                for author in np.unique(text_frame.author):
                    # If pos=True need extra string to identify graph
                    if instr[2]: 
                        pos_string = 'pos_'
                    else:
                        pos_string = ''
                    self.topic_graphs[author] = self.make_topic_graph(list(text_frame[text_frame.author == author]['graphs_'+pos_string+str(instr[0])]))
            self.topic_graphs[i] = topic_graphs_temp

        return self

    def make_graphs(self, dataframe, n, wordgram=True, pos=True):
        """Takes a dataframe of text and creates graphs of them
        """

        if (pos):
            dataframe['book'] = dataframe.apply(lambda row:BookText(rawtext=row.text).translate_to_pos(), axis=1)
            dataframe['graphs_pos_'+str(n)] = dataframe.apply(lambda row:row.book.make_graph(n, wordgram=wordgram), axis=1)
        else:
            dataframe['book'] = dataframe.apply(lambda row:BookText(rawtext=row.text))
            dataframe['graphs_'+str(n)] = dataframe.apply(lambda row:row.book.make_graph(n, wordgram=wordgram), axis=1)

    def make_topic_graph(self, topics):
    """Returns a topic graph or topic graphs dependent on the type of the argument topics

    topics (list or dict): If list, will create a topic graph from every graph in the list
                           If dict, will create a topic graph for each key. A key value pair is
                           expected to be the (key) topic name and (value) list of graphs from
                           which to create the topic graph. Will return a dictionary with the same
                           keys but who's values correspond to the topic graph
    """

    if (isinstance(topics,list)):
        # Creating empty object that will become the topic graph
        g_topic = None
        for i, graph in enumerate(topics):
            if g_topic is None:
                g_topic = graph
            else:
                gu = nx.compose(g_topic, graph)
                for edge in graph.edges:
                    if (edge not in g_topic.edges):
                        gu.edges[edge[0],edge[1]]['weight'] = graph.edges[edge[0],edge[1]]['weight']
                    elif (edge not in graph.edges):
                        gu.edges[edge[0],edge[1]]['weight'] = g_topic.edges[edge[0],edge[1]]['weight']
                    else:
                        weight_g_topic = g_topic.get_edge_data(edge[0],edge[1])['weight']
                        weight_gdi = graph.get_edge_data(edge[0],edge[1])['weight']
                        gu.edges[edge[0],edge[1]]['weight'] = weight_g_topic + (weight_gdi - weight_g_topic)/i
                g_topic = copy.deepcopy(gu)
        return g_topic

    elif (isinstance(topics,dict)):
        topic_dict = dict.fromkeys(topics.keys(),[])
        for key in topic_dict:
            graphs = topics[key]
            topic_dict[key] = make_topic_graph(graphs)
        return topic_dict

    else:
        raise ValueError("'topics' must be either a list or dict object")

    def get_graph_metrics(dataframe, topic_graphs, graph_key, col_mod):
        for i in dataframe.index:
            graph = dataframe.loc[i,graph_key]
            for j, key_j in enumerate(topic_graphs.keys()):
                author_graph = topic_graphs[key_j]
                cs = ngram_graphs.containment_similarity(author_graph, graph)
                vs = ngram_graphs.value_similarity(author_graph, graph)
                nvs = ngram_graphs.normalized_value_similarity(author_graph, graph)
                dataframe.loc[i,'cs_'+col_mod+key_j] = cs
                dataframe.loc[i,'vs_'+col_mod+key_j] = vs
                dataframe.loc[i,'nvs_'+col_mod+key_j] = nvs


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