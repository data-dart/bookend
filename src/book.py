from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize, WordNetLemmatizer
import re
import random
import string


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
            self._title = title #FIXME IN MASTER
            
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
            print ("The authors are not the same. Assigning the author of the 1st Book to the resultant bookobject")
            author = author1
        else:
            author = author1
            
        if (title1 is None and title2 is not None):
            title = title2
        elif (title1 is not None and title2 is None):
            title = title1
        elif (title1 != title2):
            print ("The authors are not the same. Assigning the title of the 1st Book to the resultant bookobject")
            title = title1
        else:
            title = title1
        return BookText(rawtext=self._text+other._text, author=author, title=title, meta=None) 

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

        garbage = '\ufeff|â€™|â€"|â€œ|â€˜|â€\x9d|â€œi|_|â€'
        cleaned = re.sub(garbage, '', cleaned)
        cleaned = cleaned.replace('-', ' ')

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
            tokens = word_tokenize(cleaned)
            cleaned = (" ").join(tokens)
            
        if deromanize:
            tokens = word_tokenize(cleaned)
            regex_roman = '^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$'
            cleaned = [word for word in tokens if re.search(
                rex_roman, word, flags=re.IGNORECASE) is None]

        if inplace:
            self._text = cleaned
        else:
            return BookText(filepath=None, rawtext=cleaned, author=self._author, title=self._title, meta=self._meta)

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

        # can't remove puncuation for sentences regardless
        if include_punctuation or 'sent' in on.lower():
            token = self._text
        else:
            # remove punctuation
            token = self._text.translate(
                str.maketrans('', '', string.punctuation + '”“’'))
        if 'word' in on.lower():
            token = word_tokenize(token.lower())
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
                token = [word for word in token if not word in stop_words]
            elif 'sent' in on.lower():
                # TODO: Is there a better way to do this?
                for i in range(len(token)):
                    sent = token[i]
                    words = sent.split()
                    words_nostop = [
                        word for word in words if not word.lower() in stop_words]
                    token[i] = (" ").join(words_nostop)
        return token

    def snippet(self, length, on, non_cont=False, inplace=False):
        """
        """
        # TODO add ability for random seed
        if 'char' in on.lower():
            assert length < len(self._text)
        elif 'word' in on.lower():
            tokens = word_tokenize(self._text)
            assert length < len(tokens)
        elif 'sent' in on.lower():
            tokens = sent_tokenize(self._text)
        else:
            raise KeyError(
                "Arugument 'on' must refer to character, word, or sentence")

        if non_cont:
            # TODO add functionality for non-continuous random sampling
            pass
        else:
            if 'char' in on.lower():
                start = random.randint(0, len(self._text) - length)
                snippet = self._text[start:start + length]
            elif 'word' in on.lower():
                start = random.randint(0, len(tokens) - length)
                snippet = (" ").join(tokens[start:start + length])
            elif 'sent' in on.lower():
                start = random.randint(0, len(tokens) - length)
                snippet = (" ").join(tokens[start:start + length])
        if inplace:
            self._text = snippet
        else:
            return BookText(rawtext=snippet, author=self.author,
                            title=self.title, meta=self.meta)
        
    def random_snippet(self, on, n_groups=1, n_on=1, start_point=1, randomized = True, with_replacement=False, rem_stopwords=False, random_seed=0):
    
        """
        link = BookText Object
        n_groups = Number of groups of snippets from a booktext object. Will return an array with 
                   n_groups number of book_text objects
        n_on = Number of characters/words/sentences in each snippet that you want.
        start_point = The point in the text from where you want to get the characters/words/sentences. Starts from 1.
        randomized = If you want the snippets picked randomly, set this to true. Else, snippets will be picked in order.
        with_replacement = (Default = False). This function will return snippets with no repeated sentences.
                            If you want repetition of sentences, set it to TRUE
        rem_stopwords = By Default, the function will not remove stopwords during tokenize. Set it to TRUE to remove
                    stopwords.
        random_seed = Set a value to get the same snippets everytime. Do not set 0, as it corresponds to using
                      system_time as seed.  
        """
        return_array = []

        if 'char' in on.lower():
            tokens = self._text
        elif 'word' in on.lower():
            tokens = self.tokenize('word', rem_stopwords)
        elif 'sent' in on.lower():
            tokens = self.tokenize('sent', rem_stopwords)
        else:
            raise KeyError(
                "Argument 'on' must refer to character, word, or sentence")
        assert n_groups*n_on < len(tokens) - start_point + 1

        if (random_seed != 0):
            random.seed(random_seed)

        for iteration in range(n_groups):
            if not randomized:
                snippet = BookText(rawtext=''.join([BookText(rawtext = tokens[num]).text + str(' ') for num in range(start_point - 1 +(iteration)*n_on,start_point - 1 +(iteration + 1)*n_on)]), author = self.author, title = self.title, meta = self.meta)
            
            else:
                random_sample_of_indices = sorted(random.sample(range(0, len(tokens) - start_point + 1), n_on), reverse=False)
                snippet = BookText(rawtext=''.join([BookText(rawtext = tokens[num]).text + str(' ') for num in random_sample_of_indices]), author = self.author, title = self.title, meta = self.meta)
                if with_replacement == False:
                    for index in sorted(random_sample_of_indices, reverse=True):
                        del tokens[index]
                if (random_seed != 0):
                        random.seed(random_seed + iteration + 1)
                        
            return_array.append(snippet)
        return return_array

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
