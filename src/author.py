from pathlib import Path
import os
import string
from src.book import BookText
import numpy as np

DATA_DIR = os.path.abspath(os.path.join('../data/raw')) 
TEST_DIR = os.path.abspath(os.path.join('../data/test'))

class Author():
    """ A class for authors """

    def __init__(self, lastname):
        """ Constructor for class Author
        
        Attributes:
        
        - name
        
        Methods:
        
        - read_all_works(self, test_data=False)
        - vocab(self, rem_stopwords=True, clean=True)
        - avg_word_length(self, rem_stopwords=True)
        - avg_words_per_sent(self)
        
        """
        
        self._name = lastname
        

    def read_all_works(self, test_data=False):
        """
        returns a list containing all books by author as BookText objects
    
        """
        data_dir = Path(DATA_DIR)
        if test_data:
            data_dir = Path(TEST_DIR)

        if not data_dir.exists():
            raise FileNotFoundError('data_dir directory does not exist')
        return [BookText(str(fle)) for fle in data_dir.iterdir() if f'{self.name}_' in str(fle)]

    def vocab(self, rem_stopwords=True, clean=True):
        """ 
        Returns total vocabulary of the author.
        
        removes stopwords by default
        cleans by default
        
        """
        all_works = self.read_all_works()
        
        book_vocab = {}
        for book in all_works:
            if rem_stopwords:
                if clean:
                    book_cleaned = book.clean(lemmatize=True, deromanize=True, lemma_pos='v', inplace=False)
                    book_vocab[book.title] = book_cleaned.tokenize(on='word', rem_stopwords=True, 
                                                           stopword_lang='english',add_stopwords=[], 
                                                           include_punctuation=False)
                else:
                    book_vocab[book.title] = book.tokenize(on='word', rem_stopwords=True, 
                                                           stopword_lang='english',add_stopwords=[], 
                                                           include_punctuation=False)
                    
                    
            else:
                if clean:
                    book_cleaned = book.clean(lemmatize=True, deromanize=True, lemma_pos='v', inplace=False)
                    book_vocab[book.title] = book_cleaned.tokenize(on='word', rem_stopwords= False, 
                                                           stopword_lang='english',add_stopwords=[], 
                                                           include_punctuation=False)
                else:
                    book_vocab[book.title] = book.tokenize(on='word', rem_stopwords=False, 
                                                       stopword_lang='english',add_stopwords=[], 
                                                       include_punctuation=False)       
 
        total_vocab = []
        for words in book_vocab.values():
            total_vocab += words

        return list(set(total_vocab))

    def avg_word_length(self, rem_stopwords=True):
        """
        returns length of an average word used by the author in all their books
        
        """
        if rem_stopwords:
            tv = self.vocab(rem_stopwords=True, clean=False)
        else:
            tv = self.vocab(rem_stopwords=False, clean=False)
            
        tot_len = 0
        for word in tv:
            tot_len += len(word)
        return np.round(tot_len/len(tv), 3)

    def avg_words_per_sent(self):
        """
        returns average words per sentence used by the author in all their books
        
        """
        all_works = self.read_all_works()
        all_sentences = []
        all_words = []
        for book in all_works:
            all_sentences += book.tokenize(on = 'sent', rem_stopwords = False)
            all_words += book.tokenize(on='word', rem_stopwords = False)
        return np.round(len(all_words)/len(all_sentences), 3)

    @property
    def name(self):
        return self._name

