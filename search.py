import argparse
import json
from bs4 import BeautifulSoup

# Import any additional libraries or modules you may need

import paths
import sqlite3
import nltk
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import numpy as np
import pandas as pd

def remove_html_tags(text):
    if text:
        soup = BeautifulSoup(text, 'html.parser')
        cleaned_text = soup.get_text()
        return cleaned_text
    else:
        return text

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Join the tokens back into a single string
    clean_text = ' '.join(tokens)
    return clean_text

def search_tv_shows(input_file, output_json_file, encoding='UTF-8'):
    try: 
        # Read the search query from the input file
        with open(input_file, encoding=encoding) as f:
            query = f.read()
        
        # Implement your search logic here to find matching TV shows
        # read from the file you saved in index.py
        # matched_shows = search_tv_shows(search_query)

        # remove html tags, punctuation, stop words and apply stemming
        query_elements = preprocess_text(remove_html_tags(query))
        # tokenize the query
        tokenized_query = word_tokenize(query_elements)
        # retrieve bm25 model
        bm25 = pickle.loads(open('index/bm25.json', 'rb').read())
        # obtain scores for the tokenized _query
        scores = bm25.get_scores(tokenized_query)

        # Get the top 3 indices
        top_indices = np.argpartition(-scores, 3)[:3]

        # Get the top 3 scores 
        top_3_scores = scores[top_indices]

        # Sort in descending order
        sorted_indices = np.argsort(-top_3_scores)
        top_3_indices_desc = top_indices[sorted_indices]
        # top_3_values_desc = top_3_scores[sorted_indices]

        # Since the index of bm25 scores does not match tvmaze_id
        # we need to retrieve the tvmaze_id based on the index of bm25 scores
        df = pd.read_csv('index/id_showname.csv')
        top3_id_list = []
        # retrieve the top 3 tvmaze_id
        for i in top_3_indices_desc:
            top3_id_list.append(df['tvmaze_id'][i])

        # get the rows that have the top3_id
        filtered_rows = df[df['tvmaze_id'].isin(top3_id_list)]

        # Convert the filtered DataFrame to a list of dictionaries
        result_list = filtered_rows[['tvmaze_id', 'showname']].to_dict(orient='records')

        # sort based on the top3_list order
        result_list_sorted = sorted(result_list, key=lambda x: top3_id_list.index(x['tvmaze_id']))

        # Write the matched shows to the output JSON file
        with open(output_json_file, 'w', encoding='UTF-8') as json_file:
            json.dump(result_list_sorted, json_file, ensure_ascii=False)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for TV shows based on a query")
    parser.add_argument("--input-file", required=True, help="Path to the input file with the search query")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for matched shows")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")

    args = parser.parse_args()

    search_tv_shows(args.input_file, args.output_json_file, args.encoding)
