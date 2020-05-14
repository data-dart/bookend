from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize, sent_tokenize, WordNetLemmatizer
import re
import random


class BookText():
    """A class for reading and manipulating texts"""

    def __init__(self, filepath=None, rawtext=None, encoding='utf-8',
                 clean=False, author=None, title=None, meta=None):
        """Constructor for BookText

        parameters:
            **NB** one of filepath or rawtext must be specified

            filepath (None): filepath to a text file to read
            rawtext (None): a raw text string
            encoding ("utf-8"): the encoding to use
            clean (False): whether to clean the text on initializing
            author (None): if not specified, inferred from text
            title (None): if not specified, inferred from text
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
        rex_start = r"START OF (THIS|THE) PROJECT GUTENBERG EBOOK (.*) \*\*\*"
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
                self._author = None
        else:
            self._author = author
        if title is None:
            try:
                self._title = re.search(
                    r"(?<=Title: )[\w\s]+(?=\n)", meta_data).group().strip()
            except AttributeError:
                self._title = None
        else:
            self._title = None

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

        garbage = '\ufeff|â€™|â€"|â€œ|â€˜|â€\x9d|â€œi|-|â€'
        cleaned = re.sub(garbage, '', cleaned)

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

        if deromanize:
            tokens = word_tokenize(cleaned)
            regex_roman = '^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$'
            cleaned = [word for word in tokens if re.search(
                rex_roman, word, flags=re.IGNORECASE) is None]

        if inplace:
            self._text = cleaned
        else:
            return BookText(filepath=None, rawtext=cleaned, author=self._author)

    def tokenize(self, on, rem_stopwords=True, stopword_lang='english',
                 add_stopwords=[]):
        """Tokenize words or sentences in the text

        Produces lists of either words or sentences contained in the text

        **NB** converts words to lower case to facilitate comparisons

        on ('word' or 'sentence'):
            whether the lists will be tokenized according to words or sentences
        rem_stopwords (default True): if stopwords should be removed from tokens
        stopword_lang (default 'english'): language of stopword corpus to use
        add_stopwords CURRENTLY UNWORKING (default []): list of words to be added to stopword list
        """
        if 'word' in on.lower():
            token = word_tokenize(self._text.lower())
        elif 'sent' in on.lower():
            token = sent_tokenize(self._text.lower())
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
                        word for word in words if not word in stop_words]
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
