import argparse
import sqlite3
import pandas as pd
import sklearn.model_selection
from bs4 import BeautifulSoup
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input, Conv1D, MaxPooling1D, Dropout
import tensorflow as tf
import keras.callbacks
import pickle
import paths
import matplotlib.pyplot as plt

# import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
# import string


def remove_html_tags(text):
    if text:
        soup = BeautifulSoup(text, 'html.parser')
        cleaned_text = soup.get_text()
        return cleaned_text
    else:
        return text
    
# def preprocess_text(text):
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Remove punctuation
#     tokens = [word for word in tokens if word not in string.punctuation]
#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word.lower() not in stop_words]
#     # Apply stemming
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(word) for word in tokens]
#     # Join the tokens back into a single string
#     clean_text = ' '.join(tokens)
#     return clean_text

def load_data(sqlite_file):
    # Connect to the SQLite database
    connection = sqlite3.connect(sqlite_file)

    # Load data into a Pandas DataFrame
    df = pd.read_sql("SELECT tvmaze.tvmaze_id, tvmaze.showname, tvmaze.description, tvmaze_genre.genre FROM tvmaze INNER JOIN tvmaze_genre ON tvmaze.tvmaze_id = tvmaze_genre.tvmaze_id", connection)

    # remove html tags
    df['description'] = df['description'].apply(remove_html_tags) 

    # remove na values from 'description' column
    df = df.dropna(subset=['description'])

    # remove empty value or length of string less than 20 from 'description' column
    df = df[(df['description'] != "") & (df['description'].str.len() >= 20)].reset_index(drop=True)

    # create genre_df with each tvmaze_id's genre encoded
    genre_df = (df.assign(count=1)
                .pivot(index='tvmaze_id', columns='genre', values='count')
                .fillna(0)
                .reset_index())
    
    # create unique_id_df where duplicated tvmaze_id are removed and the df only contains 'tvmaze_id' and 'description' columns
    unique_id_df = df.drop_duplicates(subset=['tvmaze_id']).drop(df.columns[[1,3]], axis=1)

    # merge unique_id_df and genre_df
    data = pd.merge(unique_id_df, genre_df, on='tvmaze_id', how='inner')
    connection.close()
    return data

def train_model(data):
    # only encoded values with all category columns in encoded_df
    encoded_df = data.drop(data.columns[[0,1]], axis=1)
    # Create and train a machine learning model
    target = encoded_df.values

    # for understanding setting output_sequence_length 
    data.tokenized_des = data.description.apply(word_tokenize)
    tokenized_des_count = data.tokenized_des.apply(len)
    quantile_90 = int(tokenized_des_count.quantile(0.90))

    train_val_X, test_X, train_val_y, test_y = sklearn.model_selection.train_test_split(data.description, target, test_size=0.2)
    train_X, validation_X, train_y, validation_y = sklearn.model_selection.train_test_split(train_val_X, train_val_y, test_size=0.2)
    max_tokens = 20000
    output_sequence_length = quantile_90
    embedding_dim = 64
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    vectorizer.adapt(train_X)
    inputs = Input(shape=(1,), dtype=tf.string)
    vectorized = vectorizer(inputs)
    embedded = Embedding(max_tokens + 1, embedding_dim)(vectorized)
    averaged = GlobalAveragePooling1D()(embedded)
    dropout = Dropout(0.5)(averaged)
    thinking = Dense(128, activation='relu')(dropout)
    output = Dense(encoded_df.shape[1], activation='softmax')(thinking)
    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(train_X, train_y, validation_data=(validation_X, validation_y), callbacks = callback, epochs=500)
    
    # Create histogram to understand the tokenized description count for setting output_sequence_length 
    plt.hist(tokenized_des_count, bins=30, edgecolor='k')
    # Add labels and title
    plt.xlabel('Tokenized Description Count')
    plt.ylabel('Frequency')
    plt.title('Tokenized Description Count Histogram')
    # save the histogram in a png file
    plt.savefig('tokenized_des_histogram.png')
    return model, test_X, test_y

def evaluate_model(model, test_X, test_y):
    encoded_df = data.drop(data.columns[[0,1]], axis=1)
    # Create encoded_genres.json
    genre_reverse_lookup = {i:g for (i,g) in enumerate(encoded_df.columns)}
    with open('encoded_genres.json', 'wb') as f:
        f.write(pickle.dumps(genre_reverse_lookup))
    target = encoded_df.values
    _, test_X, _, test_y = sklearn.model_selection.train_test_split(data.description, target, test_size=0.2)
    # Make predictions on the test set
    accuracy = model.evaluate(test_X, test_y)[1]
    return accuracy

def save_model(model):
    # Save the trained model to paths.location_of_model
    model.save(paths.location_of_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, help="Path to the SQLite database file")

    args = parser.parse_args()

    # Load data from the SQLite database
    data = load_data(args.training_data)

    # Train the model
    model, test_X, test_y = train_model(data)

    # Evaluate the model (assuming you want to)
    accuracy = evaluate_model(model, test_X, test_y)

    print(f"Model training complete. Accuracy: {accuracy:.2f}")

    # Do you want to retrain the model on the whole data set now?

    # Save the trained model to a file
    save_model(model)
