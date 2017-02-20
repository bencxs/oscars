# oscars
2017 Academy Awards prediction

## Introduction
Can we predict the Oscar winners with machine learning? This repo contains some endeavours towards a solution. We would like to predict the Academy Award winner for Best Picture based on historical nominations and winners in the past 16 years. 

There are two approaches for this problem.

1. Predicting the Best Picture winner based on other awards won prior (Director's Guild, Screen Writers Guild, Producer's Guild).
2. Predicting the Best Picture winner based only on the text representation of the plot synopsis for each movie title. 

Part 1 was inspired by FiveThirtyEight's analysis: [Tracking The Oscars Race](https://projects.fivethirtyeight.com/oscar-predictions-2017/), while Part 2 was inspired by hypothesising that the fundamental component of a good movie (and hence, more likely to be an Oscar winner) would essentially be the movie's plot. The ConvNet model was inspired by [Yoon Kim's architecture on sentence classification](https://github.com/yoonkim/CNN_sentence).

## Dataset & Code Files
To our knowledge, there does not exist a movie dataset which contain the features we desire, namely past awards won and the plot summary. Thus, we gathered the dataset by means of web scraping from IMDb, using the Python package: BeautifulSoup.

`movie_list.csv` - Contains the list of movie titles and release years which we would want to scrape.

### Part 1
`1_Scraper-IMDb.ipynb` - Scraper to pull in past awards data. Output file generated is: `movies.csv` for training and `movies_pred.csv` for 2017 prediction.

`2_data_exploration_model.ipynb` - Model implemented with SVM classifier.

### Part 2
`deeplearning/0_Scraper-IMDb-synopsis.ipynb` - Scraper to pull in plot synopsis. Output file is: `movies_plot.csv` for training and 
`movies_plot_pred.csv` for 2017 prediction.

`deeplearning/1_preprocess.ipynb` - Preprocessing code to tokenize words.

`deeplearning/2_model.ipynb` - Base CNN model implementation.

`deeplearning/2_model_word2vec.ipynb` - CNN model with pretrained [word2vec](https://code.google.com/p/word2vec/) weights. The pretrained word2vec binary file can be downloaded from https://code.google.com/p/word2vec/

## Libraries / Packages
* Python (2.7)
* TensorFlow
* sklearn

## Key Takeaway
* Results obtained by using past awards as the features (Part 1) are aligned with FiveThirtyEight's analysis.
* Posing the problem as an NLP classification (Part 2) is not successful thus far. The neural net does not learn the meanings of words and sentences within a plot paragraph. Using word2vec does not seem to help either. This is probably because the past winning movie titles are somewhat widely different in terms of words used (characters, environments, scenes), and hence the neural net struggles to find a general representation of a winning title. More research can be done to enable understanding of sentence-meanings within an evaluative context. Or, another approach to try is maybe to distill the natural language plot to plot structure (intro-buildup-conflict-downfall-rise-crisis-triumph) as better features.
