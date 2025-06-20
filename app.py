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
# Load data from Excel files in the data directory
def load_excel_data(filename, category):
    # Use absolute path to ensure file is found
    file_path = os.path.abspath(os.path.join('data', filename))
    print(f"Looking for file: {file_path}")
    
    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        try:
            df = pd.read_excel(file_path)
            print(f"Loaded {len(df)} rows from {filename}")
            print(f"Columns: {df.columns.tolist()}")
            
            if 'Title' in df.columns and 'Abstract' in df.columns:
                df['Category'] = category
                return df[['Title', 'Abstract', 'Category']]
            else:
                print(f"Missing required columns in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
    else:
        print(f"File not found: {file_path}")
    
    return pd.DataFrame()

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory exists: {os.path.exists('data')}")
if os.path.exists('data'):
    print(f"Files in data directory: {os.listdir('data')}")

# Load data from each Excel file
dataframes = [
    load_excel_data('ai_data.xlsx', 'Artificial Intelligence'),
    load_excel_data('distribution_data.xlsx', 'Distribution Data'),
    load_excel_data('image_processing_data.xlsx', 'Image Preprocessing'),
    load_excel_data('networking_cybersecurity_data.xlsx', 'Networking and Cybersecurity'),
    load_excel_data('se_data.xlsx', 'Software Engineering')
]

# Combine all dataframes
data = pd.concat(dataframes, ignore_index=True)
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
    # Convert probability to accuracy percentage
    accuracy = float(prob) * 100
    return jsonify({'category': pred, 'accuracy': accuracy})

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

@app.route('/insights')
def insights():
    return render_template('insights.html')

@app.route('/corpus')
def corpus():
    return render_template('corpus.html')

@app.route('/corpus_data')
def corpus_data():
    tfidf_vec = linear_pipeline.named_steps['tfidf']
    feature_names = tfidf_vec.get_feature_names_out()
    result = []
    
    # Print debug information
    print(f"Available categories in data: {data['Category'].unique()}")
    print(f"Total data rows: {len(data)}")
    print(f"Feature names count: {len(feature_names)}")
    
    for cat in FIELDS:
        cat_rows = data[data['Category'] == cat]
        print(f"Category '{cat}' has {len(cat_rows)} rows")
        
        if not cat_rows.empty:
            # Process words for this category - use actual meaningful words
            words = []
            for t in cat_rows['text']:
                if isinstance(t, str):
                    words.extend(t.split())
            
            # Get unique words and sort them
            unique_words = sorted(list(set(words)))
            
            # Calculate TF-IDF values
            tfidf_matrix = tfidf_vec.transform(cat_rows['text'])
            tfidf_avg = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            
            # Create a dictionary of word:tfidf_value pairs
            word_tfidf_pairs = []
            for idx, word in enumerate(feature_names):
                if word in unique_words:
                    # Only include words that are in this category's vocabulary
                    word_tfidf_pairs.append((word, float(tfidf_avg[idx])))
            
            # Sort by TF-IDF value in descending order to get most important terms first
            word_tfidf_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Separate into words and values for the response
            top_words = [pair[0] for pair in word_tfidf_pairs[:100]]
            top_tfidf_values = [pair[1] for pair in word_tfidf_pairs[:100]]
            
            result.append({
                'field': cat,
                'preprocessed_words': top_words,  # Most important words by TF-IDF
                'tfidf_values': top_tfidf_values  # Corresponding TF-IDF values
            })
        else:
            result.append({
                'field': cat,
                'preprocessed_words': [],
                'tfidf_values': []
            })
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
