from flask import Flask

app = Flask(__name__)

from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import request

app = Flask(__name__)
CORS(app)

nltk.data.path.append('/home/darrennat09/nltk_data')

# Load stopwords
nltk_stopwords = stopwords.words('english')

# Use stopwords

# reading model and vectorizer
with open('/home/darrennat09/mysite/svcmodel.pkl', 'rb') as filemodel:
    model = pickle.load(filemodel)

with open('/home/darrennat09/mysite/vectorizer.pkl', 'rb') as filemodel:
    tfidf = pickle.load(filemodel)

ps = PorterStemmer()

def text_preprocessing(text):
    # lowercase all characters
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # remove stopwords
    for i in text:
        if i not in nltk_stopwords:
            y.append(i)

    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route("/")
def hello():
    return "hello"

@app.route("/api/model", methods=["POST", "GET"])
def predict():
    try:
        data = request.get_json()

        text = data.get('text', '')
        if text:
            transformed_data = text_preprocessing(text)
            tfidf_data = tfidf.transform([transformed_data])
            result = model.predict(tfidf_data)[0]

            return jsonify({"result": str(result)})
        else:
            return jsonify({"error": 'no text provided'}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500