# bookend

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
