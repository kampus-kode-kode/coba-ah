from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)
Bootstrap(app)

# Load the data and model only once
df = pd.read_csv("data/names_dataset.csv")
df_X = df.name
df_Y = df.sex

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df_X)

# Loading our ML Model
with open("models/naivebayesgendermodel.pkl", "rb") as model_file:
    clf = joblib.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        namequery = request.form['namequery']
        data = [namequery]
        vect = cv.transform(data)
        my_prediction = clf.predict(vect)

        # Handle case where prediction is empty
        if my_prediction:
            return render_template('result.html', prediction=my_prediction[0], name=namequery.upper())
        else:
            return render_template('result.html', prediction='Not Found', name=namequery.upper())

if __name__ == "__main__":
    app.run(debug=True)
