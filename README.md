### <u>Introduction</u>

This project involves two parts. The first part is about creating a system classifying the genres of a television show based on the description of it. The second part creates a system that searches or displays the most likely television show name according to a description input. Both parts make use of a database containing the required information and only relevant data is extracted.

### <u>Classification</u>

- `train.py`

-  `classify.py` 

##### Classifier-design

The classifier is a model created with word embeddings. Before applying the classifier, data pre-processing is implemented. Html tags and duplicated data are removed. Also, an encoded data is created where all genre columns are presented and matching genre will have value 1, otherwise 0 for each tv show. After data pre-processing, a `png` file about tokenized description histogram is created for setting `output_sequence_length`. Then the classifier has 13 layers starting from an `Input` layer. The `Input` layer processes one string as input. Then, it comes to `TextVectorization` layer.  `TextVectorization` layer tokenizes and vectorizes the description data. After that, it is embedding layer. `Embedding` layer converts the tokens into continuous vectors and each vector will be represented based on the number of embedding dimensions. Next, there is a `GlobalAveragePooling1D` layer to performs average pooling to obtain the average value across the whole sequence for each genre. After that, a dropout layer is added to prevent overfitting. Subsequently, a dense layer is applied. Finally, it is an output layer which output the result. In the meantime, a folder `genre_model` is created to save the model for any subsequent classifications to be done with the model. Also, an `encoded_genres.json` is created for classification use. An explanation folder `explain` is created when a classification is done where it has a lime output file to show the explanation of the classification result. The top 3 genres will be displayed if they have a score higher than or equal to 5%. Otherwise, they are not displayed.

##### Reason to design this way

Word embeddings approach is chosen because we are predicting genres from the descriptions where words are from a structured space. In other words, the words have connection or share information with each other in the descriptions and word embeddings is able to capture the information. If bag-of-words is used, it might not capture those information and results in poor prediction. Transformer and pre-trained Word2Vec approaches are also possible, but in this case, we are using word embeddings. In the `TextVectorization` layer, the `output-sequence-length` is set to 90% quantile of tokenized description because it can make sure most of the tokenized descriptions can fall within the length properly without extending far. Also, it returns the best accuracy compared to other output sequence length. For the `Embedding` layer, `maxtoken +1` is to contain words/tokens that is beyond 20,000 max tokens. In terms of top 3 genres, a 5% threshold is created because 5% is a common indicator in statistics.

### <u>Search</u>

- `index.py`
- `search.py`

**Search-desgin**

The search engine make use of BM25. It is a probabilistic-based ranking function where it has a diminishing return on term frequency. Before applying BM25, we combine `showname` and `description` columns together in `index.py` and apply text pre-processing, e.g. tokenization, stemming, etc. to obtain the corpus for BM25. Then, apply BM25 on the corpus and save the model to `index` folder. Also, a dataframe with `tvmaze_id` and `showname` column is saved in `index` folder as csv file  for `search.py` use. The text pre-processing takes some time. Any search/query will now go through text pre-processing first, then processed by the BM25 model in the `index` folder and return the top 3 searches.

**Reason to design this way** 

BM25 is still having a good performance among the modern search engines. Hence, I choose this model for the search engine. Pre-process of the text is important because it enhances the accuracy of the model where all the text is standardized. The model can more easily match the search text with the corpus and return superior results. Also, it will penalise the term frequency if the same text appears for many times. The returned scores tend to provide a more balanced or fairer result as the term penalty presents. 

