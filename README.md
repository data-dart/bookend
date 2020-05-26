# bookend

bookend is a classifier for text data. We are building a machine learning model to identify the author of a work given only a selection of text. As the project progresses, we will provide further information.

This project is part of The Erdos Institute's  [data science boot camp](https://www.erdosinstitute.org/code). Data sources include [Project Gutenberg](https://www.gutenberg.org/).

## Table of contents

- [About the project](#about-the-project)
  - [Methodology](#methodology)
- [Results](#results)
- [Applications](#applications)
- [Using the code](#using-the-code)
- [References](#references)

## About the project

The goal of this project is to use the body of a text to classify it. To start, we are focusing on predicting the author of a text from the text itself. If we give our algorithm 
> It was the best of times, it was the worst of times...

will it know Charles Dickens wrote it?

### Methodology

We build many features out of the text, and use an ensemble machine learning model to predict the author.

In the future we will expand to allow any-size snippets of text and additional authors, but for now we have focused on identifying 75-sentence chunks of text. We include the following authors, chosen as the top authors on Project Gutenberg who wrote their works in English (We exclude Shakespeare, as his works are largely plays rather than novels or short stories. We additionally add JK Rowling, who famously wrote the Harry Potter series as well as crime fiction novels under the pseudonym Robert Galbraith):

| Author                | Jane Austen | Lewis Caroll | Daniel Defoe | Charles Dickens | Sir Arthur Conan Doyle | Jack London | J.K. Rowling | Mary Shelley | Robert Louis Stevenson | H.G. Wells | Oscar Wilde |
|-----------------------|-------------|--------------|--------------|-----------------|------------------------|-------------|--------------|--------------|------------------------|------------|-------------|
| Works in Training Set |             |              |              |                 |                        |             |              |              |                        |            |             |
| Work(s) Testing Set   |             |              |              |                 |                        |             |              |              |                        |            |             |
|                       |             |              |              |                 |                        |             |              |              |                        |            |             |

Below we briefly summarize several avenues we investigated for engineering features from the text. For more detail, we refer you to the [notebooks](https://github.com/data-dart/bookend/tree/master/notebooks) and [source code](https://github.com/data-dart/bookend/tree/master/src).

#### Bag-of-words

We build a "vocabulary" of known words, assembled from the unique [lemmatized](https://en.wikipedia.org/wiki/Lemmatisation) words in the corpus. For each text sample, we build a feature vector indicating how many times each vocabulary word appeared.

#### Lexical Features

These features are a grab-bag of word and sentence level statistics, like the mean and variation in word and sentence length, and the [Flesch-Kincaid grade level](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests#Flesch%E2%80%93Kincaid_grade_level).

#### Syntactic Features

These features focus on the frequency of different parts of speech used by the authors.

#### N-Grams

These features look at [n-grams](https://en.wikipedia.org/wiki/N-gram), or the frequencies with which n words appear together contiguously (for example, the phrase "how are you today?" has 2-grams "how are," "are you," and "you today.").

## Results

TBD. High accuracy. Different feature sets have different success.

## Applications

Here, we briefly discuss future applications of our model, once it is further developed.

### Plagiarism detection

### Troll (and bot) hunting

### Authorship attribution

## Using the code

Read here if you wish to reproduce our results yourself, or use some of the tools we've developed.

We recommend setting up a [conda](https://www.anaconda.com/products/individual) environment if you wish to run any of the code here. This will also install the `src` directory so it can be imported.

To clone the repo and then set up the anaconda environment, do
```
$ git clone https://github.com/data-dart/bookend.git
$ cd bookend
$ conda env create -f environment.yml
```
Then, to activate the new environment,
```
$ conda activate bookend
```

Additionally, the `nltk` package requires external data. This should be downloaded by
```
$ python
>>> import nltk
>>> nltk.download()
```
which will open a GUI that allows you to downlaod the relevant files.

For more information, or for issues relating to `nltk`, please visit https://github.com/nltk/nltk.

## References

Here are some useful links.
