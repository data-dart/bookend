import pandas as pd
import numpy as np
from .book import BookText
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
from nltk.tag import pos_tag
from nltk import FreqDist

import collections
import networkx as nx
import ngram_graphs
import copy

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

    def __init__(self, bow={}, use_passed_bow=False, include_proper_nouns=False, max_occurences=10000, min_occurences=0):
        """
        constructor for class BOWFeatures
        
        Parameters:
        
        bow: dict : of the form {word: unique int index}
        
        use_passed_bow: boolean : to use passed bow or not
        
        include_proper_nouns : boolean :  use dict with or without proper nouns. For a passed bow, this will be ignored.
        
        max_occurences: int : a word which appers more than this number of times in the corpus will be 
                                removed from the dictionary
                                
        min_occurences : int : a word which appers fewer than this number of times in the corpus 
                                will be removed from the dictionary
        
        """
        self.bow = bow
        self.use_passed_bow = use_passed_bow
        self.include_proper_nouns = include_proper_nouns
        self.max_occurences = max_occurences
        self.min_occurences = min_occurences
        

    def transform(self, X):
        """vectorizes a row of text data using the bag of words"""
        vectorizer = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                                ngram_range=(1, 1), vocabulary=self.bow)
        text_frame = _convert_to_dataframe(X)
        
         #-----------lemmatize before vectorizing--------------------------------------------   
        text_frame['text'] = text_frame['text'].apply(lambda text: BookText(rawtext=text)
                                                      .clean(lemmatize=True, deromanize=True)
                                                      .text)
        #------------------------------------------------------------------------------------
        
        X = text_frame['text'].values
        XX = vectorizer.fit_transform(X)
        
        return XX
        

    def fit(self, X, y=None):
        """Fit DataFrame by building bag of words
        
        Uses the passed bow if the flag self.use_passed_bow is True
        
        """
        text_frame = _convert_to_dataframe(X)
        
        if not self.use_passed_bow:
            self.build_bow(corpus = text_frame['text'])
            
        else:
            if self.bow =={}:
                print("Looks like you forgot to pass a Bag of words.")
                print("If you want to build a bag instead, set use_passed_bag=False .")
                
        return self

    def build_bow(self, corpus: pd.Series):
        """
        Returns a dict of unique words in the corpus

        Input:
        A corpus in the form of a pd.Series

        Output: 
        Corpus vocabulary as a dictionary.

        dict_key : word
        dict_value : an integer label for the key

        """
        #----------------------------clean, tokenize, lower--------------------------------------
        
        corpus_vals = corpus.values
        
        words = []
        for i in range(len(corpus_vals)):
            b = BookText(rawtext = corpus_vals[i])
            bb = b.clean(deromanize=True, lemmatize=True) 
            words += bb.tokenize(on='words', rem_stopwords=True)
        
        words = [w.lower() for w in words]
        
       
        
        
        #---------------------------remove too frequent and too infrequent words-----------------
        
        dist = FreqDist(words)
        repeats = {}
        for rep_count in sorted(list(set(dist.values())), reverse = True):
            if (rep_count >=self.min_occurences) & (rep_count <=self.max_occurences):
                repeats[rep_count] = [word for word in dist.keys() if dist[word] == rep_count]
                
        words_minus_tails = []
        for v in repeats.values():
            words_minus_tails += v 
        words = words_minus_tails
        
       
        

        #----------------------------remove all numeric characters ------------------------------
        no_nums = []
        for word in words:
            if (not any(ch.isdigit() for ch in word)):
                no_nums.append(word)
                
        all_words = no_nums
        
        
        
        
        #--------------------------- identify and remove all proper nouns ------------------------
        
        if self.include_proper_nouns:
            proper_nouns = []
        
        else:
            proper_nouns= self.proper_nouns_in_corpus(corpus) 
            proper_nouns = [word.lower() for word in proper_nouns]
        
        words_minus_proper = [word for word in all_words if word not in proper_nouns]
        
        #------------------------Package as a dictionary ------------------------------------------
        vocab_dict = {}
        for i in range(len(words_minus_proper)):
            vocab_dict[words_minus_proper[i]] = i
        
        self.bow = vocab_dict
        
        #------------------------------fin.------------------------------------------------------

        
        
        
    def proper_nouns_in_corpus(self, corpus: pd.Series):
        
        """ collects all proper nouns in the corpus and returns them as a list"""

        def find_proper_nouns(text):
            """ A local helper function """
            bt = BookText(rawtext=text)
            bt.clean(lemmatize=True, deromanize=True,inplace=True)
            sentences = bt.tokenize(on = 'sent') #all sentences in one row
            tagged_sentences = [pos_tag(sentence.split()) for sentence in sentences]  # tagged sentences
            p_list = [] # contains all the proper nouns in the entire row 
            for tagged_sent in tagged_sentences:
                propernouns = [word for word,pos in tagged_sent if pos == 'NNP']
                p_list += propernouns # adding all propernouns in a particular sentence
            return p_list
        
        series_temp = corpus.apply(find_proper_nouns) # apply returns a series
        
        proper_list = [] # contains all the proper nouns in the corpus
        for w in series_temp:
            proper_list += w 
            
        #----------------------------- filtering out some proper nouns------------------------------
        

        titles = ['Mr', 'Madam', 'Mrs', 'Miss', 'Sir','St', 'Lady', 
                  'Professor', 'Captain', 'King', 'Queen', 'God', 'Lord', 'Colonel', 'Master', 'Knight', 'Mayor']
        Days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        Months = ["January", "February", "March", "April", "May", 
                  "June", "July", "August", "September", "October", "November", "December"]
        Misc = ['English', 'Illustrated', 'Illustration', 'Jew', 
                'Look', 'Old', 'Thank', 'Great', 'Man', 'House', 'Death', 'Cakes', 'Square', 'Poor', 'Nay']

        proper_list_cleaned = [word.title() for word in proper_list]
        proper_list_cleaned = [word for word in proper_list_cleaned if word not in titles]
        proper_list_cleaned = [word for word in proper_list_cleaned if word not in Days]
        proper_list_cleaned = [word for word in proper_list_cleaned if word not in Months]
        proper_list_cleaned = [word for word in proper_list_cleaned if word not in Misc]
        proper_list_cleaned = [word for word in proper_list_cleaned if len(word) > 2]
        
        #---------------------------------- converting to a set------------------------------------
        
        N_repeats = 0 
        # optional parameter which decides how many times a proper noun needs 
        # to occur in the corpus for it to be included in the list of proper nouns.

        dist = FreqDist(proper_list_cleaned)
        
        repeats = {}
        for rep_count in sorted(list(set(dist.values())), reverse = True):
            if rep_count >=N_repeats:
                repeats[rep_count] = [word for word in dist.keys() if dist[word] == rep_count]
                
        proper_nouns = []
        for v in repeats.values():
            proper_nouns += v
        
        return proper_nouns
            
                
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
            pos_unigram = pos_as_arr
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
        
# This has not been tested yet        

class NGramFeatures(BaseEstimator, TransformerMixin):
    """Converts text into n-gram features"""

    def __init__(self, ngrams=[(1,True,True),(2,True,True),(3,True,True)]):
        """ngram_instr is a list of tuples with instructions on how to create the ngram graphs and features
           format: [(n_1, wordgram_1, pos_1),(n_2, wordgram_2, pos_2),...]
           example: If you wanted uni-word-grams using POS tagged text you would pass ngrams=[(1, True, True)]
        """
        self.ngrams = ngrams

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
        for column in transformed_X.columns:
            if (('graph' in column) or ('author' in column) or ('text' in column) or ('book' in column)):
                transformed_X.drop(columns=column, inplace=True)
        return transformed_X

    def fit(self, X, y=None, random_seed=None, tdsplit=0.5):
        # builds the n-gram graph or graphs
        text_frame = _convert_to_dataframe(X)
        # Creating an empty array to store the dictionaries of topic graphs later
        self.topic_graphs = np.empty(len(self.ngrams),dtype='object')
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
                    topic_graphs_temp[author] = self.make_topic_graph(list(text_frame[y == author]['graphs_'+pos_string+str(instr[0])]))
            else:
                for author in np.unique(text_frame.author):
                    # If pos=True need extra string to identify graph
                    if instr[2]: 
                        pos_string = 'pos_'
                    else:
                        pos_string = ''
                    topic_graphs_temp[author] = self.make_topic_graph(list(text_frame[text_frame.author == author]['graphs_'+pos_string+str(instr[0])]))
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

    def get_graph_metrics(self, dataframe, topic_graphs, graph_key, col_mod):
        """Takes in a dataframe and adds to it columns for each similarity metric
        """
        for i in dataframe.index:
            graph = dataframe.loc[i,graph_key]
            for j, key_j in enumerate(topic_graphs.keys()):
                author_graph = topic_graphs[key_j]
                cs = ngram_graphs.containment_similarity(author_graph, graph)
                vs = ngram_graphs.value_similarity(author_graph, graph)
                nvs = ngram_graphs.normalized_value_similarity(author_graph, graph)
                dataframe.loc[i,'cs_'+col_mod+str(key_j)] = cs
                dataframe.loc[i,'vs_'+col_mod+str(key_j)] = vs
                dataframe.loc[i,'nvs_'+col_mod+str(key_j)] = nvs

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
