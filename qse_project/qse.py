import PyPDF2
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
from rouge import Rouge
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# Function to answer user questions

def answer_question(source_text, user_question, threshold=0.2):

    print('source_text: ', source_text, '\nuser_question: ', user_question)

    # Preprocess the text (this function should be defined based on your preprocessing needs)
    def preprocess(text):
        # Preprocess text (fine-tuning)
        return text

    # Assuming 'source_text' and 'user_question' are defined and 'threshold' is set
    text = source_text['text']
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocess(text)] + [preprocess(user_question)])
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    print(cosine_similarities)
    # Find the highest score
    max_score_index = np.argmax(cosine_similarities)
    print("cosine_similarity Score Index: ",max_score_index)
    max_score = cosine_similarities[max_score_index]
    print('max_score:', max_score)

    # Check if the score is above the threshold
    if max_score > threshold:
        sentences = text.split('.')
        # Return the sentence with the highest cosine similarity score
        return sentences[max_score_index].strip()
    else:
        return "No answer found from the given input"


# Summarisation Module
def summarise_module(input_text):
    print(input_text)
    tokenizer_t5 = T5Tokenizer.from_pretrained("t5-base")
    model_t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
    inputs = tokenizer_t5.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model_t5.generate(inputs, max_length=500, min_length=40, length_penalty=2.0, num_beams=4,
                                    early_stopping=True)

    # Decode and print the summary
    summary_t5 = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)

    model_name = "facebook/bart-large-cnn"
    tokenizer_bart = BartTokenizer.from_pretrained(model_name)
    model_bart = BartForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer_bart.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary using BART
    summary_ids = model_bart.generate(input_ids, max_length=500, num_beams=4, length_penalty=1.0, early_stopping=True)

    summary_bart = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    rouge = Rouge()
    scores_t5 = rouge.get_scores(summary_t5, input_text)
    scores_bart = rouge.get_scores(summary_bart, input_text)

    # Choose the summary with the higher ROUGE-1 F1 score
    if scores_t5[0]['rouge-1']['f'] > scores_bart[0]['rouge-1']['f']:
        return summary_t5
    else:
        return summary_bart


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    global input_text
    input_text = request.json

    print(input_text)
    print(input_text['text'], type(input_text['text']))

    txt = input_text['text']
    summary = summarise_module(txt)

    # print("Generated Summary: ",summary)
    rouge = Rouge()
    accuracy_score = rouge.get_scores(txt, summary)
    print("Accuracy Scores between the input-text and the summarised text: \n", accuracy_score)
    # return jsonify()
    response_data = {'summary': summary}
    return jsonify(response_data)

# summarize-pdf
@app.route('/summarizepdf', methods=['POST'])
def summarize_pdf():
    # This route should handle the PDF summarization logic
    file = request.files['file']
    print(file, type(file))
    # Implement PDF summarization logic here
    # For example, read the file and extract text for summarization
    summary = "This is where the summary of the PDF would go."
    return jsonify({'summary': summary})
    # global input_text
    # # Assuming 'pdf_path' is the key in the JSON payload
    # pdf_path = request.json
    # print(pdf_path, type(pdf_path))
    # input_text = ""
    # with open(pdf_path, 'rb') as file:
    #     pdf_reader = PyPDF2.PdfReader(file)
    #     for page in pdf_reader.pages:
    #         input_text += page.extract_text()
    #
    # print("input text:\n", input_text, type(input_text))
    #
    # # Assuming 'summarise_module' is a function that takes text and returns a summary
    # summary = summarise_module(input_text)
    #
    # rouge = Rouge()
    # accuracy_score = rouge.get_scores(input_text, summary)
    # print("Accuracy Scores between the input-text and the summarised text: \n", accuracy_score)
    #
    # response_data = {'summary': summary}
    # return jsonify(response_data)

#
@app.route('/answer', methods=['POST'])
def chatbot():
    print("chat_bot")
    input_question = request.json
    threshold = request.json
    # print(input_question, type(input_question))
    question = input_question['text']
    print("Entered Question: ",question)
    answer = answer_question(input_text, question)
    print("Provided Answer: ",answer)

    return jsonify({'answer': answer})


if __name__ == "__main__":
    app.run(debug=True)
