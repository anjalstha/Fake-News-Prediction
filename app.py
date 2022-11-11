from flask import Flask, request, render_template
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import nltk
nltk.download('stopwords')

df = pd.read_csv('train.csv')

df = df.fillna('')

df['content'] = df['author']+' '+df['title']

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

df['content'] = df['content'].apply(stemming)

X = df['content']
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

vectorizer = TfidfVectorizer()
XV_train = vectorizer.fit_transform(X_train)
XV_test = vectorizer.transform(X_test)

lr = LogisticRegression()
lr.fit(XV_train, Y_train)

svm = svm.SVC(kernel='linear', gamma='auto')
svm.fit(XV_train, Y_train)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def prediction():

    news = request.form['news']

    testing_news = {"data":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["data"] = new_def_test["data"].apply(stemming)
    new_x_test = new_def_test["data"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    pred_svm = svm.predict(new_xv_test)

    def output_lable(n):
        if n == 1:
            return "Fake"
        elif n == 0:
            return "True"

    pred = output_lable(pred_lr[0]),output_lable(pred_svm[0])


    return render_template("prediction.html", data=pred)

if __name__ == "__main__":
    app.run(debug=True)
