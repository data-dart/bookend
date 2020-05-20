from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize, WordNetLemmatizer
import re
import random
import string
import numpy as np
from warnings import warn


class BookText():
    """A class for reading and manipulating texts"""

    def __init__(self, filepath=None, rawtext=None, encoding='utf-8', file_format='standard',
                 clean=False, author=None, title=None, meta=None, infer_toc=True):
        """Constructor for BookText
        parameters:
            **NB** one of filepath or rawtext must be specified
            filepath (None): filepath to a text file to read
            rawtext (None): a raw text string
            encoding ("utf-8"): the encoding to use
            file_format ("standard"): format of the filepath string, used to assign author and title information if it cannot be found in the meta data
            clean (False): whether to clean the text on initializing
            author (None): if not specified, inferred from text
            title (None): if not specified, inferred from text
            infer_toc (True): will attempt to infer the TOC from the text
        """
        if not (filepath or rawtext):
            raise ValueError('Must specify one of filepath or rawtext')
        if filepath is not None:
            # Reading in the file
            with open(filepath, encoding=encoding) as f:
                data = f.read()
        else:
            data = rawtext
        # Using regex to mark the start and end of the book
        rex_start = r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK (.*)\s*\*\*\*"
        rex_end = r"(?i)END of (this|the|) Project Gutenberg"
        try:
            start_pos = re.search(rex_start, data).span()[1]
        except AttributeError:  # re.search returned None 
            start_pos = 0
        try:
            end_pos = re.search(rex_end, data).span()[0]
        except AttributeError:  # re.search returned None 
            end_pos = None
        meta_data = data[:start_pos]
        text_of_book = data[start_pos:end_pos]

        if infer_toc:
            toc_start, toc_end = self.find_toc(text_of_book)
            meta_data = meta_data + text_of_book[:toc_start]
            self._toc = text_of_book[toc_start:toc_end]
            text_of_book = text_of_book[toc_end:]
        else:
            self._toc = None


        self._text = text_of_book
        if clean:
            self.clean(inplace=True)
        self.start_pos = start_pos
        self.end_pos = end_pos
        self._meta = meta or meta_data
        if author is None:
            try:
                self._author = re.search(
                    r"(?<=Author: )[\w\s]+(?=\n)", meta_data).group().strip()
            except AttributeError:
                if file_format == 'standard' and filepath is not None:
                    self._author = filepath.split('/')[-1].split('.')[0].split('_')[0]
                else:
                    self._author = None
        else:
            self._author = author
        if title is None:
            try:
                self._title = re.search(
                    r"(?<=Title: )[\w\s]+(?=\n)", meta_data).group().strip()
            except AttributeError:
                if file_format == 'standard' and filepath is not None:
                    self._title = filepath.split('/')[-1].split('.')[0].split('_')[1]
                else:
                    self._title = None
        else:
            self._title = title

            
    def __add__(self, other): 
        '''
        Overloaded the addition operator for BookText. Returns text of both the booktext objects, and carries over
        the author and title info. If authors or titles are not same, the 1st book is preferred over the second.
        '''
        author1 = self._author
        author2 = other._author
        title1 = self._title
        title2 = other._title
        if (author1 is None and author2 is not None):
            author = author2
        elif (author1 is not None and author2 is None):
            author = author1
        elif (author1 != author2):
            warn("The authors are not the same. Assigning the author of the 1st Book to the resultant bookobject")
            author = author1
        else:
            author = author1
            
        if (title1 is None and title2 is not None):
            title = title2
        elif (title1 is not None and title2 is None):
            title = title1
        elif (title1 != title2):
            warn("The titles are not the same. Assigning the title of the 1st Book to the resultant bookobject")
            title = title1
        else:
            title = title1
        return BookText(rawtext=self._text+' '+other._text, author=author, title=title, meta=None) 


    def clean(self, lemmatize=True, deromanize=False, lemma_pos='v', inplace=False):
        """Cleans the full text
        Returns a new BookText object with cleaned text derived 
        from the full text
        lemmatize (default True): cleaned data will be lemmatized
        deromanize (default False): roman numerals will be removed from cleaned data
        lemma_pos (default 'v'): position variable for lemmatizer
            if lemma_pos='all', will run through all lemmatization types for each word
        inplace (default False): replaces the text of the object with cleaned text
        """
        cleaned = self._text

        garbage = '\ufeff|â€™|â€"|â€œ|â€˜|â€\x9d|â€œi|_|â€|£|éé|ô|à|â|ê|—£|éè|ü|é|œ|î|æ|ç|‘|é—|…|ö|è'
        
        #TODO: substitute ligature characters properly using  https://en.wikipedia.org/wiki/List_of_words_that_may_be_spelled_with_a_ligature
        
        cleaned = re.sub(garbage, '', cleaned)
        cleaned = cleaned.replace('-', ' ')
        cleaned = re.sub(r'\n+', ' ', cleaned)
        cleaned = cleaned.replace('-', ' ').replace('—', ' ')

        if lemmatize:
            WNLemma = WordNetLemmatizer()
            # The Lemmatizer only takes single words as inputs, so we need to
            # break our text into individual words
            tokens = word_tokenize(cleaned)
            if lemma_pos != 'all':
                lemmatized = [WNLemma.lemmatize(
                    token, pos=lemma_pos) for token in tokens]
            else:
                lemmatized = [WNLemma.lemmatize(WNLemma.lemmatize(
                    WNLemma.lemmatize(token, pos='n'), pos='v'), pos='a')
                    for token in tokens]
            cleaned = (" ").join(lemmatized)
        else:
            # Added this bit just to clean up unnecessary spaces in the text.
            # The Clean Function thus returns a text with no extra spaces.
            cleaned = re.sub('\s+', ' ', cleaned)
            
        if deromanize:
            tokens = word_tokenize(cleaned)
            regex_roman = '^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$'
            deromanized = [word for word in tokens if re.search(
                regex_roman, word, flags=re.IGNORECASE) is None]
            cleaned = (" ").join(deromanized)

        if inplace:
            self._text = cleaned
        else:
            return BookText(filepath=None, rawtext=cleaned, author=self.author, title=self.title, meta=self.meta)
            
            

    def tokenize(self, on, rem_stopwords=True, stopword_lang='english',
                 add_stopwords=[], include_punctuation=False):
        """Tokenize words or sentences in the text
        Produces lists of either words or sentences contained in the text
        **NB** word tokenize converts words to lower case to facilitate comparisons
        on ('word' or 'sentence'):
            whether the lists will be tokenized according to words or sentences
        rem_stopwords (default True): if stopwords should be removed from tokens
        stopword_lang (default 'english'): language of stopword corpus to use
        add_stopwords CURRENTLY UNWORKING (default []): list of words to be added to stopword list
        """

        # can't remove puncuation prior to splitting sentences
        # because we need the periods, question marks, and the like
        if include_punctuation or 'sent' in on.lower():
            token = self._text
        else:
            # remove punctuation
            token = self._text.translate(
                str.maketrans('', '', string.punctuation + '”“’'))
        if 'word' in on.lower():
            token = word_tokenize(token) #Changed here from token.lower() to token
        elif 'sent' in on.lower():
            token = sent_tokenize(token)
        else:
            raise KeyError(
                "Arugument 'on' must refer to either word or sentence")
        if rem_stopwords:
            stop_words = set(stopwords.words(stopword_lang))
            # Additional words can be added to the stop word list
            # stop_words.extend(add_stopwords)
            if 'word' in on.lower():
                token = [word for word in token if not word.lower() in stop_words]
            elif 'sent' in on.lower():
                # TODO: Is there a better way to do this?
                for i in range(len(token)):
                    sent = token[i]
                    words = sent.split()
                    words_nostop = [
                        word for word in words if not word.lower() in stop_words]
                    token[i] = (" ").join(words_nostop)
        # if tokenizing by sentence and removing punctuation
        # now that we've split sentences (i.e. on periods)
        # we can iterate through tokens and remove punctuation
        if 'sent' in on.lower() and not include_punctuation:
            for i in range(len(token)):
                token[i] = token[i].translate(
                    str.maketrans('', '', string.punctuation + '”“’'))
        return token

    def snippet(self, length, on, groups=1, non_cont=False, with_replacement=True,
                rem_stopwords=False, randomized=True, ret_as_arr=None, 
                random_seed=None, inplace=False):
        """
        Returns snippets of char/words/sent of various length depending on input.

        length: the length of the snippet, in units of on
        on: whether to divide based on characters ('char'), 
                    words ('word'), or sentences ('sent')
        groups (1): number of separate snippets to return
        non_cont (False): if True, separate snippets are not-continuous, and rather are sampled
        with_replacement (True): if non_cont, then determines whether snippets can be repeated.
                                 Otherwise, does nothing
        randomized (True): if False and non_cont is False, then 
                    starts dividing snippets from the beginning of the text. Otherwise, picks
                    a random starting point. Does nothing if non_cont is True, since snippets
                    are already selected randomly
        ret_as_array: (None): If True, returns array of BookText objects, one for each group
                              If False, returns a single BookText object with all snippets combined
                              If None (default): does True if non_cont is True,
                                                 and False if not_cont is False.
                                                 or, if groups = 1, does False regardless
        random_seed (None): random seed passed to numpy
        rem_stopword (False): passed to the tokenize functions
        inplace (False): if True, replace text with output instead of returning. Only
                         allowed if ret_as_arr is False
        """
        
        #Random seed set if provided by user.
        if (random_seed is not None):
            random.seed(random_seed)

        # if None, takes on same value as non_cont
        if ret_as_arr is None:
            ret_as_arr = non_cont
            allow_size_one_array = False
        else:
            allow_size_one_array = ret_as_arr

        if ret_as_arr and inplace:
            raise ValueError('Cannot assign an array of text as self._text')
        
        #Punctuations are now retained in a snippet.
        if 'char' in on.lower():
            tokens = self._text
            # join_string determines how we glue tokens back together
            join_string = ''
        elif 'word' in on.lower():
            tokens = self.tokenize('word', rem_stopwords=rem_stopwords, include_punctuation=True)
            join_string = ' '
        elif 'sent' in on.lower():
            tokens = self.tokenize('sent', rem_stopwords=rem_stopwords, include_punctuation=True)
            join_string = ' '
        else:
            raise KeyError(
                "Argument 'on' must refer to character, word, or sentence")
        
        #Groups*Length needs to be lower than the tokenized text length
        if groups * length > len(tokens):
            raise ValueError("Can't request more snippets than there is text:\n"
                             "Reduce groups and/or length")
        return_array = []
        
        #Random number generated for starting point. Everything before starting point
        #is removed from array
        
        if non_cont:
            # sample starting indices (one for each group)
            indices = np.random.choice(np.arange(len(tokens) - length + 1), replace=with_replacement,
                                       size=groups)
        else:
            if randomized:
                start = random.randint(0, len(tokens) - groups*length)
                tokens = tokens[start:]
            # there are [groups] indices, spaced length apart
            indices = np.arange(0, groups) * length
        return_array = [BookText(rawtext=join_string.join(tokens[ind:ind + length]),
                                 author=self.author, title=self.title, meta=self.meta) 
                                 for ind in indices]

        # if not returning as array, or if groups is 1 and it was unspecified
        if (groups == 1 and not allow_size_one_array):
            snip = return_array[0]
        elif ret_as_arr:
            snip = np.array(return_array)
        else:
            # thanks to the __add__ method above, this works
            snip = np.sum(return_array)
        
        if inplace and not ret_as_arr:
            self._text = snip._text
        else:
            return snip

    def word_count(self, unique=False, **kwargs):
        """Returns a count of the words in the BookText object
        If unique, returns only the number of unique words
        See tokenize for possible keyword arguments
        """
        token = self.tokenize(on='word', **kwargs)
        if unique:
            token = set(token)
        return len(token)

    def sentence_count(self, **kwargs):
        """Returns a count of the sentences in the BookText object
        See tokenize for possible keyword arguments
        """
        token = self.tokenize(on='sent', **kwargs)
        return len(token)

    def features(self, ):
        # TODO create function to return feature vectors such as:
        # vector of most common words
        # sparse matrix of most frequently used words in the english language
        # unique words
        pass

    @property
    def meta(self):
        """Returns the meta data"""
        return self._meta

    @property
    def text(self):
        """Return the full text of the book"""
        return self._text

    @property
    def author(self):
        return self._author

    @property
    def title(self):
        return self._title

    @staticmethod
    def find_toc(text, toc_reg='\n\s+(table of |.?)contents.?\s+\n'):
        """Returns start and end indices of TOC"""
        try:
            ind_start = re.search(toc_reg, text.lower()).span()[1]
        except AttributeError:
            return (0, 0)
        # TOC styles are too varied to do this exactly, so we make a guess
        ind_stop = ind_start + text[ind_start:].find('\n\n\n')
        toc_text = text[ind_start:ind_stop]
        main_text = text[ind_stop:]
        return ind_start, ind_stop

    @property
    def toc(self):
        return self._toc