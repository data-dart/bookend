import pytest
from src.book import BookText


def test_read_file():
    booktest = BookText('example_text.txt')
    assert 1


def test_read_rawtext():
    booktest = BookText(rawtext='Easy raw text')
    assert 1


@pytest.fixture
def bt():
    bookobj = BookText('example_text.txt')
    return bookobj


def test_toc(bt):
    assert bt.toc.startswith('Chap 1')
    assert bt.toc.endswith('Chap 3')


def test_scrape_meta(bt):
    assert bt.author == 'Testy McTestface'


def test_infer_meta(bt):
    assert bt.title == 'text'


def test_word_token_remstop_no_punct(bt):
    assert len(bt.tokenize('word', rem_stopwords=True,
                           include_punctuation=False)) == 7


def test_word_token_remstop_yes_punct(bt):
    assert len(bt.tokenize('word', rem_stopwords=True,
                           include_punctuation=True)) == 14


def test_word_token_no_punct(bt):
    assert len(bt.tokenize('word', rem_stopwords=False,
                           include_punctuation=False)) == 19


def test_word_token_yes_punct(bt):
    assert len(bt.tokenize('word', rem_stopwords=False,
                           include_punctuation=True)) == 26


def test_snippet_on_sent(bt):
    snip = bt.snippet(2, 'sent')
    assert len(snip.tokenize('sent')) == 2


def test_snippet_on_word(bt):
    snip = bt.snippet(4, 'word')
    assert (len(snip.tokenize('word', rem_stopwords=False,
                             include_punctuation=True)) == 4)


def test_snippet_char_continuity(bt):
    snip = bt.snippet(1, 'char', groups=10, non_cont=True)
    assert snip.text.replace(' ', '') not in bt.text.replace(' ', '')
    snip = bt.snippet(1, 'char', groups=10, non_cont=False)
    assert snip.text.replace(' ', '') in bt.text.replace(' ', '')