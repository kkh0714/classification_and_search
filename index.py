import argparse
import sqlite3
import json
import os

# Import any additional libraries or modules you may need
import pickle
from bs4 import BeautifulSoup
import pandas as pd
import paths
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import string

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

def index_data(sqlite_file):
    try:
        # Connect to the SQLite database
        connection = sqlite3.connect(sqlite_file)
        df = pd.read_sql("SELECT showname, COALESCE(description, 'no description') AS description FROM tvmaze", connection)
        # remove html tags of the description data
        df['description'] = df['description'].apply(remove_html_tags)
        # combine description and showname columns
        df['showname_description'] = df['showname'].str.cat(df['description'], sep=' ')
        # preprocess the showname_description data
        df['showname_description'] = df['showname_description'].apply(preprocess_text)

        # tokenize the showname_description data
        tokenized_corpus = [word_tokenize(doc) for doc in df['showname_description']]

        # using bm25 to process the corpus
        bm25 = BM25Okapi(tokenized_corpus)

        # Save the indexed data to a separate file (e.g., JSON? Flat lines?)
        os.makedirs('index', exist_ok=True)
        with open(os.path.join(paths.location_of_index, 'bm25.json'), 'wb') as index_file:
            index_file.write(pickle.dumps(bm25))
        # Save the dataframe in csv format for search use
        id_showname = pd.read_sql("SELECT tvmaze_id, showname FROM tvmaze", connection)
        id_showname.to_csv(os.path.join(paths.location_of_index, 'id_showname.csv'), index=False)

        connection.close()
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index data from an SQLite database")
    parser.add_argument("--raw-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()

    index_data(args.raw_data)
