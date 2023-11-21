import argparse
import json
import os

# Import any additional libraries or modules you may need
import keras
import pickle
import paths
import pandas as pd
import lime.lime_text
from bs4 import BeautifulSoup

# def preprocess_text(text):
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Remove punctuation
#     tokens = [word for word in tokens if word not in string.punctuation]
#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word.lower() not in stop_words]
#     # Apply stemming
#     tokens = [stemmer.stem(word) for word in tokens]
#     # Join the tokens back into a single string
#     clean_text = ' '.join(tokens)
#     return clean_text

def remove_html_tags(text):
    if text:
        soup = BeautifulSoup(text, 'html.parser')
        cleaned_text = soup.get_text()
        return cleaned_text
    else:
        return text

def classify_tv_show(input_file, output_json_file, encoding='UTF-8', explanation_output_dir=None):
    try:
        # Read the description from the input file
        encoding = 'utf-8'
        with open(input_file, encoding=encoding) as f:
            description = f.read()
        
        description = remove_html_tags(description)
        # Load your model, perhaps from paths.location_of_model
        inference_model = keras.models.load_model(paths.location_of_model)
        
        # Implement your classification logic here to identify TV show genres
        # load your model from somewhere in the /app directory
        inference_genre_lookup = pickle.loads(open('encoded_genres.json', 'rb').read())
        raw_prediction = inference_model.predict([description])
        prediction_frame = pd.DataFrame(data=raw_prediction, columns=inference_genre_lookup.values())
        # Only keep genres greater than or equal to 5%
        filtered_df = prediction_frame[prediction_frame >= 0.05]
        # sort the top 3 genres in descending order
        top_3_genres = filtered_df.stack().sort_values(ascending=False)[:3].index.get_level_values(1).tolist()

        # Write the identified genres to the output JSON file
        with open(output_json_file, 'w', encoding='UTF-8') as json_file:
            json.dump(top_3_genres, json_file, ensure_ascii=False)

        # Optionally, write an explanation to the explanation output directory
        genre_encoded_list = []
        genre_list = []
        for key, value in inference_genre_lookup.items():
            genre_encoded_list.append(key)
            genre_list.append(value)

        text_explainer = lime.lime_text.LimeTextExplainer(class_names=genre_list, bow=False)
        lime_explanation = text_explainer.explain_instance(description, inference_model.predict, labels=genre_encoded_list)
        # explanation_output_dir = os.path.join('.', 'explain')

        if explanation_output_dir:
            if not os.path.exists(explanation_output_dir):
                os.makedirs(explanation_output_dir)

            lime_filename = os.path.join(explanation_output_dir, "lime_explanation.html")

            # with open(lime_filename, 'w', encoding='UTF-8') as html_file:
            #     lime_explanation.save_to_file(lime_filename)
            lime_explanation.save_to_file(lime_filename)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify TV show genres based on description")
    parser.add_argument("--input-file", required=True, help="Path to the input file with TV show description")
    parser.add_argument("--output-json-file", required=True, help="Path to the output JSON file for genres")
    parser.add_argument("--encoding", default="UTF-8", help="Input file encoding (default: UTF-8)")
    parser.add_argument("--explanation-output-dir", help="Directory for explanation output")

    args = parser.parse_args()

    input_file_path = args.input_file

    with open(input_file_path, 'r') as file:
        content = file.read()

    output_file_path = args.output_json_file

    explanation_file_path = args.explanation_output_dir

    classify_tv_show(args.input_file, args.output_json_file, args.encoding, args.explanation_output_dir)

