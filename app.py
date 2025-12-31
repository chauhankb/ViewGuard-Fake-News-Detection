from flask import Flask, render_template, request
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    processed_text = preprocess_text(news)
    vector = vectorizer.transform([processed_text])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "FAKE NEWS ❌"
        color = "red"
    else:
        result = "REAL NEWS ✅"
        color = "green"

    return render_template(
        "index.html",
        prediction=result,
        result_color=color,
        news_text=news
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

