import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Thesis fields
FIELDS = [
    'Artificial Intelligence',
    'Distribution Data',
    'Image Preprocessing',
    'Networking and Cybersecurity',
    'Software Engineering'
]

# --- Preprocessing Function ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# --- Load Real Data for Thesis Fields ---
data = pd.read_csv('preprocessed_papers_data.csv')
data = data.dropna(subset=['Title', 'Abstract', 'Category'])
data['text'] = data['Title'].astype(str) + ' ' + data['Abstract'].astype(str)
data['text'] = data['text'].apply(preprocess)

# Label Encoding
y = data['Category']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split for metrics
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- Pipelines ---
linear_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', MaxAbsScaler()),
    ('svm', SVC(kernel='linear', probability=True, random_state=42))
])
poly_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', MaxAbsScaler()),
    ('svm', SVC(kernel='poly', degree=3, probability=True, random_state=42))
])

# Fit both
linear_pipeline.fit(X_train, y_train)
poly_pipeline.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html', fields=FIELDS)

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    text = preprocess(title + ' ' + abstract)
    kernel = request.form.get('kernel', 'linear')
    if kernel == 'poly':
        pred_code = poly_pipeline.predict([text])[0]
        prob = np.max(poly_pipeline.predict_proba([text]))
    else:
        pred_code = linear_pipeline.predict([text])[0]
        prob = np.max(linear_pipeline.predict_proba([text]))
    pred = label_encoder.inverse_transform([pred_code])[0]
    return jsonify({'category': pred, 'prob': round(float(prob), 2)})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Compare linear and polynomial
    y_pred_linear = linear_pipeline.predict(X_test)
    y_pred_poly = poly_pipeline.predict(X_test)
    report_linear = classification_report(
        y_test, y_pred_linear, target_names=label_encoder.classes_, output_dict=True)
    report_poly = classification_report(
        y_test, y_pred_poly, target_names=label_encoder.classes_, output_dict=True)
    return jsonify({
        'linear': report_linear,
        'poly': report_poly
    })

@app.route('/tfidf_values', methods=['GET'])
def tfidf_values():
    # Show TF-IDF matrix (first 10 papers for brevity)
    tfidf_vec = linear_pipeline.named_steps['tfidf']
    tfidf_matrix = tfidf_vec.transform(X_train[:10])
    feature_names = tfidf_vec.get_feature_names_out()
    dense = tfidf_matrix.todense().tolist()
    return jsonify({'features': feature_names.tolist(), 'matrix': dense})

@app.route('/preprocessed_data', methods=['GET'])
def preprocessed_data():
    # Show preprocessed text and labels (first 10)
    return jsonify({
        'data': data[['text', 'Category']].head(10).to_dict(orient='records')
    })

@app.route('/paper_count', methods=['GET'])
def paper_count():
    return jsonify({'count': len(data)})

if __name__ == '__main__':
    app.run(debug=True)
