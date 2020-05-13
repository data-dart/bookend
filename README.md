# bookend

bookend is a classifier for text data. We are building a machine learning model to identify the author of a work given only a selection of text. As the project progresses, we will provide further information.

This project is part of The Erdos Institute's  [data science boot camp](https://www.erdosinstitute.org/code). Data sources include [Project Gutenberg](https://www.gutenberg.org/).

## Getting started

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
