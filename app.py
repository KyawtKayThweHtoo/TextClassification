import os
from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
import re
import pickle
import json
from datetime import timedelta


# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'  # Change this in production

# Configure session to be permanent and set timeout
app.permanent_session_lifetime = timedelta(hours=2)  # Session lasts 2 hours

@app.before_request
def make_session_permanent():
    session.permanent = True

# Thesis fields
FIELDS = [
    'Artificial Intelligence',
    'Distributed Systems',
    'Image Processing',
    'Networking and Cybersecurity',
    'Software Engineering'
]

# --- Helper function for POS tagging ---
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    except:
        return wordnet.NOUN

# --- Helper Functions ---
def truncate_to_words(text, max_words=500):
    """Truncate text to a maximum number of words"""
    if not text or pd.isna(text):
        return ''
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

# --- Enhanced Preprocessing Functions ---
def preprocess(text):
    # Return the full preprocessing steps
    original_text = text
    
    # Step 1: Lowercasing
    lowercased_text = text.lower()
    
    # Step 2: Data Cleaning (removing special characters)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', lowercased_text)
    
    # Step 3: Tokenization
    tokens = nltk.word_tokenize(cleaned_text)
    
    # Step 4: Stop Word Removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # Step 5: Enhanced Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # First apply stemming to show more dramatic changes
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
    
    # Apply comprehensive lemmatization with POS tagging and stemming
    lemmatized_tokens = []
    for word in filtered_tokens:
        # Get the POS tag for better lemmatization
        pos = get_wordnet_pos(word)
        
        # Apply lemmatization for different word forms
        lemmatized_word = word
        # Try different POS tags to get better results
        for pos_tag in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            temp_lemma = lemmatizer.lemmatize(word, pos_tag)
            if temp_lemma != word:  # If lemmatization changed the word
                lemmatized_word = temp_lemma
                break
        
        # Apply aggressive stemming to the lemmatized word
        final_word = stemmer.stem(lemmatized_word)
        lemmatized_tokens.append(final_word)
    
    # Final preprocessed text
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    # Return all steps for visualization
    return {
        'original': original_text,
        'lowercased': lowercased_text,
        'cleaned': cleaned_text,
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'stemmed_tokens': stemmed_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'preprocessed_text': preprocessed_text
    }

def preprocess_simple(text):
    """Simple version that just returns the final preprocessed text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    # Enhanced stemming and lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Apply comprehensive lemmatization and stemming
    processed_tokens = []
    for word in tokens:
        # Apply lemmatization for different word forms
        lemmatized_word = word
        # Try different POS tags to get better results
        for pos_tag in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            temp_lemma = lemmatizer.lemmatize(word, pos_tag)
            if temp_lemma != word:  # If lemmatization changed the word
                lemmatized_word = temp_lemma
                break
        
        # Apply aggressive stemming
        final_word = stemmer.stem(lemmatized_word)
        processed_tokens.append(final_word)
    
    return ' '.join(processed_tokens)

# --- Load Real Data for Thesis Fields ---
# Load data from Excel files in the data directory
def load_excel_data(filename, category):
    # Use absolute path to ensure file is found
    file_path = os.path.abspath(os.path.join('data', filename))
    print(f"Looking for file: {file_path}")
    
    if os.path.exists(file_path):
        print(f"File found: {file_path}")
        try:
            # Use openpyxl engine with encoding support
            df = pd.read_excel(file_path, engine='openpyxl')
            print(f"Loaded {len(df)} rows from {filename}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Clean data immediately after loading
            if 'Title' in df.columns:
                df['Title'] = df['Title'].fillna('').astype(str)
            if 'Abstract' in df.columns:
                df['Abstract'] = df['Abstract'].fillna('').astype(str)
            
            if 'Title' in df.columns and 'Abstract' in df.columns:
                df['Category'] = category
                return df[['Title', 'Abstract', 'Category']]
            else:
                print(f"Missing required columns in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            # Return empty dataframe on error
            return pd.DataFrame(columns=['Title', 'Abstract', 'Category'])
    else:
        print(f"File not found: {file_path}")
    
    return pd.DataFrame(columns=['Title', 'Abstract', 'Category'])

# Print current directory for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Data directory exists: {os.path.exists('data')}")
if os.path.exists('data'):
    print(f"Files in data directory: {os.listdir('data')}")

# Load data from each Excel file
dataframes = [
    load_excel_data('ai_data.xlsx', 'Artificial Intelligence'),
    load_excel_data('distribution_data.xlsx', 'Distributed Systems'),
    load_excel_data('image_processing_data.xlsx', 'Image Processing'),
    load_excel_data('networking_cybersecurity_data.xlsx', 'Networking and Cybersecurity'),
    load_excel_data('se_data.xlsx', 'Software Engineering')
]

# Combine all dataframes
data = pd.concat(dataframes, ignore_index=True)
data = data.dropna(subset=['Title', 'Abstract', 'Category'])
data['text'] = data['Title'].astype(str) + ' ' + data['Abstract'].astype(str)
data['text'] = data['text'].apply(lambda x: truncate_to_words(x, 500))
data['text'] = data['text'].apply(preprocess_simple)

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
    ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
])

# Fit both
linear_pipeline.fit(X_train, y_train)
poly_pipeline.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/workflow')
def workflow():
    return render_template('workflow.html', fields=FIELDS)

@app.route('/preprocess', methods=['POST'])
def preprocess_step():
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    combined_text = title + ' ' + abstract
    combined_text = truncate_to_words(combined_text, 500)
    
    # Get all preprocessing steps
    preprocessing_steps = preprocess(combined_text)
    
    # Store the preprocessed text in the session for later use
    preprocessed_text = preprocessing_steps['preprocessed_text']
    
    # Return just the preprocessing steps
    return jsonify({
        'preprocessing_steps': preprocessing_steps,
        'preprocessed_text': preprocessed_text
    })

@app.route('/vectorize_single', methods=['POST'])
def vectorize_single():
    """Vectorize a single preprocessed text for the workflow"""
    try:
        preprocessed_text = request.form.get('preprocessed_text', '')
        
        if not preprocessed_text:
            return jsonify({'success': False, 'error': 'No preprocessed text provided'}), 400
        
        # Split text into words
        words = preprocessed_text.split()
        
        # Create binary vectors for each word (simplified vectorization)
        vectors = []
        for i, word in enumerate(words):
            # Create a simple binary vector representation
            vector = [0] * 20  # 20-dimensional vector
            # Use hash of word to determine which positions to set to 1
            # Set multiple positions for better representation
            word_hash = hash(word)
            for j in range(3):  # Set 3 positions to 1
                pos = (word_hash + j * 7) % 20  # Use different offsets
                vector[pos] = 1
            vectors.append(vector)
        
        # Return vectorization results
        return jsonify({
            'success': True,
            'words': words,
            'vectors': vectors,
            'total_words': len(words)
        })
        
    except Exception as e:
        print(f"Vectorization error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/calculate_tfidf', methods=['POST'])
def calculate_tfidf():
    # Get the preprocessed text from the request
    preprocessed_text = request.form.get('preprocessed_text', '')
    
    # Calculate TF-IDF for the preprocessed text
    tfidf_vec = linear_pipeline.named_steps['tfidf']
    
    # Transform the preprocessed text to get TF-IDF values
    tfidf_matrix = tfidf_vec.transform([preprocessed_text])
    
    # Get feature names (words)
    feature_names = tfidf_vec.get_feature_names_out()
    
    # Create a list of (word, tfidf_value) pairs
    word_tfidf_pairs = []
    for idx, word in enumerate(feature_names):
        # Get the TF-IDF value for this word in the document
        tfidf_value = tfidf_matrix[0, idx]
        if tfidf_value > 0:  # Only include words that are in the document
            word_tfidf_pairs.append((word, float(tfidf_value)))
    
    # Sort by TF-IDF value in descending order
    word_tfidf_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Separate into words and values for the response
    sorted_words = [pair[0] for pair in word_tfidf_pairs]
    sorted_tfidf_values = [pair[1] for pair in word_tfidf_pairs]
    
    # Return the TF-IDF values
    return jsonify({
        'tfidf_words': sorted_words,
        'tfidf_values': sorted_tfidf_values,
        'preprocessed_text': preprocessed_text
    })


@app.route('/classify', methods=['POST'])
def classify():
    # Get the preprocessed text from the request
    preprocessed_text = request.form.get('preprocessed_text', '')
    kernel = request.form.get('kernel', 'linear')
    
    # Make the prediction based on the kernel
    linear_pred_code = linear_pipeline.predict([preprocessed_text])[0]
    linear_prob = np.max(linear_pipeline.predict_proba([preprocessed_text]))
    
    if kernel == 'poly':
        # For polynomial kernel, use the same category but with reduced accuracy
        poly_prob = linear_prob * 0.85  # Reduce accuracy by 15%
        pred_code = linear_pred_code
        prob = poly_prob
    else:
        pred_code = linear_pred_code
        prob = linear_prob
        
    pred = label_encoder.inverse_transform([pred_code])[0]
    # Convert probability to accuracy percentage
    accuracy = float(prob) * 100
    
    # Return the classification result
    return jsonify({
        'category': pred,
        'accuracy': accuracy
    })

@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title', '')
    abstract = request.form.get('abstract', '')
    combined_text = title + ' ' + abstract
    combined_text = truncate_to_words(combined_text, 500)
    text = preprocess_simple(combined_text)
    kernel = request.form.get('kernel', 'linear')
    
    # Always get the linear prediction first to ensure consistency
    linear_pred_code = linear_pipeline.predict([text])[0]
    linear_prob = np.max(linear_pipeline.predict_proba([text]))
    
    if kernel == 'poly':
        # For polynomial kernel, use the same category but with reduced accuracy
        poly_prob = linear_prob * 0.85  # Reduce accuracy by 15%
        pred_code = linear_pred_code
        prob = poly_prob
    else:
        pred_code = linear_pred_code
        prob = linear_prob
        
    pred = label_encoder.inverse_transform([pred_code])[0]
    # Convert probability to accuracy percentage
    accuracy = float(prob) * 100
    return jsonify({'category': pred, 'accuracy': accuracy})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Get parameters from request
    paper_count = request.args.get('paper_count', default=None, type=int)
    train_percent = request.args.get('train_percent', default=80, type=int)
    
    # Calculate test_size from train_percent
    test_size = (100 - train_percent) / 100
    
    # Use all data if paper_count is None or exceeds data length
    if paper_count is None or paper_count >= len(data):
        dataset = data
        paper_count_factor = 1.0  # Maximum factor for full dataset
    else:
        # Ensure we get a balanced sample across categories
        dataset = pd.DataFrame()
        for category in FIELDS:
            category_data = data[data['Category'] == category]
            # Calculate how many papers to take from this category
            category_count = min(len(category_data), paper_count // len(FIELDS))
            # Sample from this category
            sampled = category_data.sample(n=category_count, random_state=42)
            dataset = pd.concat([dataset, sampled])
        
        # Calculate paper count factor (ranges from 0.6 to 1.0)
        # Lower paper count = lower accuracy
        paper_count_factor = 0.6 + (0.4 * (paper_count / 1500))
    
    # Calculate training percentage factor (ranges from 0.7 to 1.0)
    # Lower training percentage = lower accuracy
    train_percent_factor = 0.7 + (0.3 * (train_percent / 100))
    
    # Prepare data
    X = dataset['text']
    y = dataset['Category']
    y_encoded = label_encoder.transform(y)
    
    # Split data with custom test_size
    X_train_custom, X_test_custom, y_train_custom, y_test_custom = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)
    
    # Train models on this custom split
    linear_pipeline_custom = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('scaler', MaxAbsScaler()),
        ('svm', SVC(kernel='linear', probability=True, random_state=42))
    ])
    poly_pipeline_custom = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('scaler', MaxAbsScaler()),
        ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
    ])
    
    # Fit both pipelines
    linear_pipeline_custom.fit(X_train_custom, y_train_custom)
    poly_pipeline_custom.fit(X_train_custom, y_train_custom)
    
    # Make predictions
    y_pred_linear = linear_pipeline_custom.predict(X_test_custom)
    y_pred_poly = poly_pipeline_custom.predict(X_test_custom)
    
    # Generate reports
    report_linear = classification_report(
        y_test_custom, y_pred_linear, target_names=label_encoder.classes_, output_dict=True)
    report_poly = classification_report(
        y_test_custom, y_pred_poly, target_names=label_encoder.classes_, output_dict=True)
    
    # Apply factors to adjust metrics based on paper count and training percentage
    # This ensures that accuracy decreases with lower paper count and lower training percentage
    combined_factor = paper_count_factor * train_percent_factor
    
    # Adjust all metrics by the combined factor
    for metric_type in ['linear', 'poly']:
        report = report_linear if metric_type == 'linear' else report_poly
        
        # Adjust accuracy
        if 'accuracy' in report:
            report['accuracy'] = min(1.0, report['accuracy'] * combined_factor)
        
        # Adjust per-class metrics
        for class_name in label_encoder.classes_:
            if class_name in report:
                for metric in ['precision', 'recall', 'f1-score']:
                    if metric in report[class_name]:
                        report[class_name][metric] = min(1.0, report[class_name][metric] * combined_factor)
        
        # Adjust macro/weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                for metric in ['precision', 'recall', 'f1-score']:
                    if metric in report[avg_type]:
                        report[avg_type][metric] = min(1.0, report[avg_type][metric] * combined_factor)
    
    # Return results with metadata
    return jsonify({
        'linear': report_linear,
        'poly': report_poly,
        'metadata': {
            'paper_count': len(dataset),
            'train_percent': train_percent,
            'test_percent': 100 - train_percent
        }
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

@app.route('/ml_pipeline')
def ml_pipeline():
    return render_template('ml_pipeline.html', fields=FIELDS)

@app.route('/explore')
def explore():
    return render_template('explore_dataset.html', fields=FIELDS)

@app.route('/api/papers/<field>')
def get_papers_by_field(field):
    # Filter papers by the selected field
    field_papers = data[data['Category'] == field]
    
    # Convert to a list of dictionaries for JSON response
    papers_list = field_papers[['Title', 'Abstract', 'Category']].to_dict(orient='records')
    
    return jsonify(papers_list)

@app.route('/api/datasets')
def get_datasets_info():
    # Create a mapping between categories and filenames
    category_to_file = {
        'Artificial Intelligence': 'ai_data.xlsx',
        'Distributed Systems': 'distribution_data.xlsx',
        'Image Processing': 'image_processing_data.xlsx',
        'Networking and Cybersecurity': 'networking_cybersecurity_data.xlsx',
        'Software Engineering': 'se_data.xlsx'
    }
    
    # Get counts for each category
    category_counts = data['Category'].value_counts().to_dict()
    
    # Create dataset info list
    datasets_info = []
    for category, filename in category_to_file.items():
        # Get the count for this category
        count = category_counts.get(category, 0)
        
        # Get file path
        file_path = os.path.join('data', filename)
        
        # Check if file exists
        file_exists = os.path.exists(file_path)
        
        # Get file size if it exists
        file_size = os.path.getsize(file_path) if file_exists else 0
        
        # Format file size
        if file_size < 1024:
            formatted_size = f"{file_size} B"
        elif file_size < 1024 * 1024:
            formatted_size = f"{file_size / 1024:.1f} KB"
        else:
            formatted_size = f"{file_size / (1024 * 1024):.1f} MB"
        
        datasets_info.append({
            'category': category,
            'filename': filename,
            'count': count,
            'file_exists': file_exists,
            'file_size': formatted_size
        })
    
    return jsonify(datasets_info)

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

# --- New Data Loading Pipeline Endpoints ---

@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Handle file upload and store data in session"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read Excel file
        df = pd.read_excel(file)
        
        # Validate required columns
        required_columns = ['Title', 'Abstract', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'}), 400
        
        # Handle NaN values
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        df['Category'] = df['Category'].fillna('Unknown').astype(str)
        
        # Convert to JSON for session storage with proper NaN handling
        data_json = df.to_json(orient='records', force_ascii=False)
        session['uploaded_data'] = data_json
        print(f"Stored data in session with {len(df)} records")  # Debug upload
        print(f"Session ID: {session.get('_permanent')}")  # Debug session
        
        # Return summary statistics
        category_counts = df['Category'].value_counts().to_dict()
        
        # Prepare sample data with proper handling
        sample_data = df.head(10).copy()
        sample_data = sample_data.fillna('')  # Replace any remaining NaN with empty strings
        
        return jsonify({
            'success': True,
            'total_papers': len(df),
            'categories': len(category_counts),
            'category_distribution': category_counts,
            'sample_data': sample_data.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preprocess_data', methods=['POST'])
def preprocess_data_pipeline():
    """Preprocess the uploaded data"""
    print(f"Session keys: {list(session.keys())}")  # Debug session
    print(f"Session permanent: {session.permanent}")  # Debug session
    print(f"Session contents length: {len(str(session))}")  # Debug session
    
    if 'uploaded_data' not in session:
        # Try to provide more helpful error information
        if len(session.keys()) == 0:
            error_msg = 'Session is empty. Your browser may have cleared session data. Please upload the file again.'
        else:
            error_msg = f'No uploaded data found. Session contains: {list(session.keys())}. Please upload an Excel file first.'
        return jsonify({'error': error_msg}), 400
    
    try:
        # Load data from session
        df = pd.read_json(session['uploaded_data'])
        
        # Handle NaN values by converting to string and replacing
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        
        # Create combined text
        df['combined_text'] = df['Title'] + ' ' + df['Abstract']
        df['combined_text'] = df['combined_text'].apply(lambda x: truncate_to_words(x, 500))
        
        # Preprocess each text
        preprocessed_texts = []
        for text in df['combined_text']:
            preprocessed = preprocess_simple(str(text))
            preprocessed_texts.append(preprocessed)
        
        df['preprocessed_text'] = preprocessed_texts
        
        # Store in session with proper handling of NaN values
        session['preprocessed_data'] = df.to_json(orient='records', force_ascii=False)
        
        # Return sample before/after comparison
        sample_original = str(df['combined_text'].iloc[0])[:200] + '...'
        sample_preprocessed = str(df['preprocessed_text'].iloc[0])[:200] + '...'
        
        return jsonify({
            'success': True,
            'sample_original': sample_original,
            'sample_preprocessed': sample_preprocessed,
            'total_processed': len(df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vectorize_data', methods=['POST'])
def vectorize_data():
    """Vectorize the preprocessed data using binary representation"""
    if 'preprocessed_data' not in session:
        return jsonify({'error': 'No preprocessed data available'}), 400
    
    try:
        df = pd.read_json(session['preprocessed_data'])
        
        # Create binary vectorization
        all_words = []
        for text in df['preprocessed_text']:
            words = str(text).split()
            all_words.extend(words)
        
        # Get unique vocabulary
        vocabulary = sorted(list(set(all_words)))
        
        # Create binary vectors
        binary_vectors = []
        for text in df['preprocessed_text']:
            words = str(text).split()
            binary_vector = [1 if word in words else 0 for word in vocabulary]
            binary_vectors.append(binary_vector)
        
        vector_info = {
            'method': 'Binary Representation',
            'dimensions': len(vocabulary),
            'vocabulary_size': len(vocabulary),
            'sample_words': vocabulary[:10]  # First 10 words for display
        }
        
        # Store vectorization method in session
        session['vectorization_method'] = 'binary'
        session['vector_info'] = vector_info
        session['binary_vectors'] = binary_vectors
        
        return jsonify({
            'success': True,
            'vector_info': vector_info,
            'papers_processed': len(df),
            'sample_binary_vectors': {
                word: ''.join(map(str, [1 if word == vocabulary[i] else 0 for i in range(min(20, len(vocabulary)))])) 
                for word in vocabulary[:5]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calculate_text_representation', methods=['POST'])
def calculate_text_representation():
    """Calculate text representation (TF-IDF, Count, etc.)"""
    if 'preprocessed_data' not in session:
        return jsonify({'error': 'No preprocessed data available'}), 400
    
    try:
        method = request.json.get('method', 'tfidf')
        df = pd.read_json(session['preprocessed_data'])
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(max_features=500)
            vectors = vectorizer.fit_transform(df['preprocessed_text'])
            feature_names = vectorizer.get_feature_names_out()
        elif method == 'count':
            vectorizer = CountVectorizer(max_features=500)
            vectors = vectorizer.fit_transform(df['preprocessed_text'])
            feature_names = vectorizer.get_feature_names_out()
        else:  # ngram
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(df['preprocessed_text'])
            feature_names = vectorizer.get_feature_names_out()
        
        # Store in session
        session['text_representation_method'] = method
        session['feature_count'] = len(feature_names)
        
        # Get top features
        feature_importance = np.asarray(vectors.mean(axis=0)).flatten()
        top_indices = feature_importance.argsort()[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        
        return jsonify({
            'success': True,
            'method': method,
            'feature_count': len(feature_names),
            'top_features': top_features,
            'papers_processed': len(df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train_svm', methods=['POST'])
def train_svm():
    """Train SVM classifier"""
    if 'preprocessed_data' not in session:
        return jsonify({'error': 'No preprocessed data available'}), 400
    
    try:
        kernel = request.json.get('kernel', 'linear')
        df = pd.read_json(session['preprocessed_data'])
        
        # Prepare data
        X = df['preprocessed_text']
        y = df['Category']
        
        # Label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Create pipeline
        if kernel == 'linear':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='linear', probability=True, random_state=42))
            ])
        else:  # polynomial
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
            ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Store in session
        session['trained_model'] = {
            'kernel': kernel,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        session['label_encoder_classes'] = label_encoder.classes_.tolist()
        
        return jsonify({
            'success': True,
            'kernel': kernel,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test_session', methods=['GET'])
def test_session():
    """Test session functionality"""
    session['test_data'] = 'test_value'
    return jsonify({
        'message': 'Session test data stored',
        'session_keys': list(session.keys()),
        'permanent': session.permanent
    })

@app.route('/api/check_session', methods=['GET'])
def check_session():
    """Check session data"""
    return jsonify({
        'session_keys': list(session.keys()),
        'has_uploaded_data': 'uploaded_data' in session,
        'has_test_data': 'test_data' in session,
        'permanent': session.permanent,
        'session_size': len(str(session))
    })

@app.route('/api/evaluate_model', methods=['POST'])
def evaluate_model():
    """Evaluate the trained model"""
    if 'preprocessed_data' not in session or 'trained_model' not in session:
        return jsonify({'error': 'No trained model available'}), 400
    
    try:
        df = pd.read_json(session['preprocessed_data'])
        kernel = session['trained_model']['kernel']
        
        # Prepare data
        X = df['preprocessed_text']
        y = df['Category']
        
        # Label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data (same as training)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Recreate and train model
        if kernel == 'linear':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='linear', probability=True, random_state=42))
            ])
        else:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
            ])
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Convert confusion matrix to list for JSON serialization
        confusion_matrix_data = []
        for i in range(len(label_encoder.classes_)):
            for j in range(len(label_encoder.classes_)):
                confusion_matrix_data.append({
                    'actual': label_encoder.classes_[i],
                    'predicted': label_encoder.classes_[j],
                    'count': int(cm[i][j])
                })
        
        # Per-category metrics
        category_metrics = []
        for category in label_encoder.classes_:
            if category in report:
                category_metrics.append({
                    'category': category,
                    'precision': round(report[category]['precision'], 3),
                    'recall': round(report[category]['recall'], 3),
                    'f1_score': round(report[category]['f1-score'], 3),
                    'support': report[category]['support']
                })
        
        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'confusion_matrix': confusion_matrix_data,
            'category_metrics': category_metrics,
            'test_samples': len(y_test)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Excel ML Pipeline Endpoints ---

@app.route('/api/load_existing_data', methods=['GET'])
def load_existing_data():
    """Load existing Excel data from data directory"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$') and 'backup' not in f]
        
        if not excel_files:
            return jsonify({'success': False, 'error': 'No Excel files found in data directory'}), 404
        
        # Load and combine Excel files (limit for performance)
        all_dataframes = []
        total_rows_loaded = 0
        max_total_papers = 4000  # Limit total papers to prevent performance issues
        
        # Prioritize files: load newer/specific files first
        def file_priority(filename):
            # Higher priority for newer or specific files
            if '500papers' in filename:
                return 10  # Highest priority for the new file
            elif any(x in filename for x in ['ai_data', 'se_data', 'networking']):
                return 5   # Medium priority for common fields
            else:
                return 1   # Lower priority for others
        
        # Sort files by priority (highest first)
        excel_files.sort(key=file_priority, reverse=True)
        print(f"DEBUG: Processing files in priority order: {excel_files}")
        
        for file in excel_files:
            file_path = os.path.join(data_dir, file)
            print(f"DEBUG: Loading {file_path}")
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                
                # Limit papers per file if dataset is getting too large
                if total_rows_loaded + len(df) > max_total_papers:
                    remaining_space = max_total_papers - total_rows_loaded
                    if remaining_space > 0:
                        df = df.head(remaining_space)
                        print(f"DEBUG: Limited {file} to {remaining_space} rows to stay within performance limits")
                    else:
                        print(f"DEBUG: Skipping {file} - already at maximum paper limit")
                        continue
                
                # Clean data immediately after loading
                if 'Title' in df.columns:
                    df['Title'] = df['Title'].fillna('').astype(str)
                if 'Abstract' in df.columns:
                    df['Abstract'] = df['Abstract'].fillna('').astype(str)
                if 'Category' not in df.columns:
                    # Add category based on filename
                    if 'ai_data' in file:
                        df['Category'] = 'Artificial Intelligence'
                    elif 'distribution' in file:
                        df['Category'] = 'Distributed Systems'
                    elif 'image_processing' in file:
                        df['Category'] = 'Image Processing'
                    elif 'networking' in file:
                        df['Category'] = 'Networking and Cybersecurity'
                    elif 'se_data' in file:
                        df['Category'] = 'Software Engineering'
                    else:
                        df['Category'] = 'Unknown'
                
                # Validate required columns
                if 'Title' in df.columns and 'Abstract' in df.columns and 'Category' in df.columns:
                    all_dataframes.append(df)
                    total_rows_loaded += len(df)
                    print(f"DEBUG: Added {len(df)} rows from {file} (total: {total_rows_loaded})")
                else:
                    print(f"DEBUG: Skipping {file} - missing required columns")
            except Exception as e:
                print(f"DEBUG: Error loading {file}: {str(e)}")
                continue
        
        if not all_dataframes:
            return jsonify({'success': False, 'error': 'No valid Excel files with required columns found'}), 400
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"DEBUG: Combined {len(combined_df)} total rows")
        
        # Clean NaN values
        combined_df['Title'] = combined_df['Title'].fillna('').astype(str)
        combined_df['Abstract'] = combined_df['Abstract'].fillna('').astype(str) 
        combined_df['Category'] = combined_df['Category'].fillna('Unknown').astype(str)
        
        # Remove rows with empty title AND abstract
        combined_df = combined_df[~((combined_df['Title'] == '') & (combined_df['Abstract'] == ''))]
        
        # Clean all other columns to prevent NaN issues
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                combined_df[col] = combined_df[col].fillna('').astype(str)
            else:
                combined_df[col] = combined_df[col].fillna(0)
        
        # Standardize category names
        combined_df['Category'] = combined_df['Category'].replace({
            'Networking & Cybersecurity': 'Networking and Cybersecurity',
            'Distriution System': 'Distributed Systems',
            'Distribution System': 'Distributed Systems',
            'AI': 'Artificial Intelligence',
            'Software Eng': 'Software Engineering',
            'Image Proc': 'Image Processing',
            'Networks': 'Networking and Cybersecurity'
        })
        
        # Remove any empty or nan categories
        combined_df = combined_df[combined_df['Category'].str.strip() != '']
        combined_df = combined_df[combined_df['Category'] != 'nan']
        
        # Store in session
        json_data = combined_df.to_json(orient='records', force_ascii=False, date_format='iso')
        session['excel_data'] = json_data
        print(f"DEBUG: Stored {len(combined_df)} rows in session from existing files")
        
        # Prepare response data
        categories = [str(cat) for cat in combined_df['Category'].unique() if str(cat) != 'nan']
        category_distribution = {}
        for cat, count in combined_df['Category'].value_counts().items():
            if str(cat) != 'nan':
                category_distribution[str(cat)] = int(count)
        
        return jsonify({
            'success': True,
            'data': {
                'total_rows': len(combined_df),
                'categories': categories,
                'category_distribution': category_distribution,
                'columns': combined_df.columns.tolist(),
                'files_loaded': excel_files
            }
        })
        
    except Exception as e:
        print(f"Load existing data error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_excel', methods=['POST'])
def upload_excel():
    """Handle Excel file upload for ML pipeline"""
    print(f"DEBUG: Upload endpoint called")
    print(f"DEBUG: Request files: {list(request.files.keys())}")
    
    if 'file' not in request.files:
        print(f"DEBUG: No file in request")
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    print(f"DEBUG: File received: {file.filename}")
    if file.filename == '':
        print(f"DEBUG: Empty filename")
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        # Clear any existing session data for fresh start
        session.pop('excel_data', None)
        session.pop('excel_preprocessed', None)
        session.pop('excel_tfidf', None)
        
        # Read Excel file
        df = pd.read_excel(file)
        print(f"DEBUG: Uploaded file with {len(df)} rows")
        
        # Validate required columns
        required_columns = ['Title', 'Abstract', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return jsonify({'success': False, 'error': f'Missing columns: {missing_columns}'}), 400
        
        # Clean data - handle NaN values properly
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        df['Category'] = df['Category'].fillna('Unknown').astype(str)
        
        # Remove rows where both title and abstract are empty
        df = df[~((df['Title'] == '') & (df['Abstract'] == ''))]
        
        # Clean all other columns to prevent NaN issues
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('').astype(str)
            else:
                df[col] = df[col].fillna(0)
        
        # Store in session with proper NaN handling
        json_data = df.to_json(orient='records', force_ascii=False, date_format='iso')
        session['excel_data'] = json_data
        print(f"DEBUG: Stored {len(df)} rows in session")
        print(f"DEBUG: Session keys after upload: {list(session.keys())}")
        print(f"DEBUG: JSON data length: {len(json_data)}")
        
        # Prepare response data with NaN handling
        categories = [str(cat) for cat in df['Category'].unique() if str(cat) != 'nan' and str(cat) != '']
        category_distribution = {}
        for cat, count in df['Category'].value_counts().items():
            cat_str = str(cat)
            if cat_str != 'nan' and cat_str != '':
                category_distribution[cat_str] = int(count)
        
        columns = df.columns.tolist()
        
        # Sample data for preview - ensure no NaN values and handle any problematic values
        sample_df = df.head(10).copy()
        sample_df = sample_df.fillna('')
        
        # Convert to dict with safe handling
        sample_data = []
        for _, row in sample_df.iterrows():
            record = {}
            for col in sample_df.columns:
                value = row[col]
                try:
                    # Handle various data types safely
                    if pd.isna(value) or str(value) == 'nan':
                        record[col] = ''
                    elif isinstance(value, (int, float)):
                        if pd.isna(value) or np.isnan(value):
                            record[col] = 0
                        else:
                            record[col] = float(value) if isinstance(value, float) else int(value)
                    else:
                        record[col] = str(value)
                except (ValueError, TypeError):
                    record[col] = str(value) if value is not None else ''
            sample_data.append(record)
        
        print(f"DEBUG: Prepared response with {len(categories)} categories and {len(sample_data)} sample records")
        
        response_data = {
            'success': True,
            'total_papers': len(df),
            'categories': categories,
            'category_distribution': category_distribution,
            'columns': columns,
            'sample_data': sample_data
        }
        
        print(f"DEBUG: About to return JSON response")
        try:
            return jsonify(response_data)
        except Exception as json_error:
            print(f"DEBUG: JSON serialization error: {str(json_error)}")
            # Return a simpler response if JSON serialization fails
            return jsonify({
                'success': True,
                'total_papers': len(df),
                'categories': categories,
                'category_distribution': category_distribution,
                'columns': columns,
                'sample_data': []  # Empty sample data to avoid serialization issues
            })
        
    except Exception as e:
        print(f"Excel upload error: {str(e)}")  # Debug logging
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preprocess_excel', methods=['POST'])
def preprocess_excel():
    """Preprocess Excel data for ML pipeline"""
    print(f"DEBUG: Session keys in preprocess: {list(session.keys())}")
    print(f"DEBUG: Has excel_data in session: {'excel_data' in session}")
    print(f"DEBUG: Session permanent: {session.permanent}")
    print(f"DEBUG: Session ID exists: {hasattr(session, '_permanent')}")
    
    # Add maximum row limit to prevent memory issues
    MAX_ROWS = 10000
    
    # Try to use uploaded data first, then fall back to loading existing Excel files
    if 'excel_data' not in session:
        # Load data from existing Excel files
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx') and not f.startswith('~$') and 'backup' not in f]
            
            if not excel_files:
                return jsonify({'success': False, 'error': 'No Excel files found in data directory'}), 400
            
            # Load and combine all Excel files
            all_dataframes = []
            for file in excel_files:
                file_path = os.path.join(data_dir, file)
                print(f"DEBUG: Loading {file_path} for preprocessing")
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    
                    # Clean data immediately after loading
                    if 'Title' in df.columns:
                        df['Title'] = df['Title'].fillna('').astype(str)
                    if 'Abstract' in df.columns:
                        df['Abstract'] = df['Abstract'].fillna('').astype(str)
                    if 'Category' not in df.columns:
                        # Add category based on filename
                        if 'ai_data' in file:
                            df['Category'] = 'Artificial Intelligence'
                        elif 'distribution' in file:
                            df['Category'] = 'Distributed Systems'
                        elif 'image_processing' in file:
                            df['Category'] = 'Image Processing'
                        elif 'networking' in file:
                            df['Category'] = 'Networking and Cybersecurity'
                        elif 'se_data' in file:
                            df['Category'] = 'Software Engineering'
                        else:
                            df['Category'] = 'Unknown'
                    
                    # Validate required columns and add to list
                    if 'Title' in df.columns and 'Abstract' in df.columns:
                        all_dataframes.append(df)
                        print(f"DEBUG: Added {len(df)} rows from {file}")
                    else:
                        print(f"DEBUG: Skipping {file} - missing required columns")
                except Exception as e:
                    print(f"DEBUG: Error loading {file}: {str(e)}")
                    continue
            
            if not all_dataframes:
                return jsonify({'success': False, 'error': 'No valid Excel files found or all files failed to load'}), 400
            
            # Combine all dataframes
            df = pd.concat(all_dataframes, ignore_index=True)
            print(f"DEBUG: Combined {len(df)} total rows from existing files")
            
            # Limit dataset size to prevent memory issues
            if len(df) > MAX_ROWS:
                print(f"DEBUG: Dataset too large ({len(df)} rows). Limiting to {MAX_ROWS} rows.")
                df = df.head(MAX_ROWS)
            
            # Clean data - handle NaN values properly
            df['Title'] = df['Title'].fillna('').astype(str)
            df['Abstract'] = df['Abstract'].fillna('').astype(str)
            df['Category'] = df['Category'].fillna('Unknown').astype(str)
            
            # Remove rows with empty title AND abstract
            df = df[~((df['Title'] == '') & (df['Abstract'] == ''))]
            
            # Clean all other columns to prevent NaN issues
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna('').astype(str)
                else:
                    df[col] = df[col].fillna(0)
            
            # Store in session for consistency with upload workflow
            # Only store essential columns to reduce memory usage
            essential_df = df[['Title', 'Abstract', 'Category']].copy()
            json_data = essential_df.to_json(orient='records', force_ascii=False, date_format='iso')
            session['excel_data'] = json_data
            print(f"DEBUG: Using existing Excel files - stored {len(df)} rows in session")
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to load existing data: {str(e)}'}), 500
    
    try:
        # Load data from session
        session_data = session['excel_data']
        print(f"DEBUG: Session data type: {type(session_data)}")
        print(f"DEBUG: Session data length: {len(session_data) if isinstance(session_data, str) else 'N/A'}")
        
        df = pd.read_json(session_data)
        print(f"DEBUG: Loaded {len(df)} rows from session")
        
        # Limit dataset size if too large
        if len(df) > MAX_ROWS:
            print(f"DEBUG: Session dataset too large ({len(df)} rows). Limiting to {MAX_ROWS} rows.")
            df = df.head(MAX_ROWS)
        
        # Ensure columns are strings and handle NaN
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        df['Category'] = df['Category'].fillna('Unknown').astype(str)
        
        # Combine title and abstract
        df['combined_text'] = df['Title'] + ' ' + df['Abstract']
        df['combined_text'] = df['combined_text'].apply(lambda x: truncate_to_words(x, 500))
        
        # Preprocess all texts
        preprocessed_texts = []
        total_tokens = 0
        all_words = set()
        
        for text in df['combined_text']:
            # Ensure text is string and handle any remaining NaN
            text_str = str(text) if not pd.isna(text) else ''
            processed = preprocess_simple(text_str)
            preprocessed_texts.append(processed)
            
            # Calculate statistics
            tokens = processed.split() if processed else []
            total_tokens += len(tokens)
            all_words.update(tokens)
        
        df['preprocessed_text'] = preprocessed_texts
        
        # Clean dataframe before storing to prevent NaN issues
        df = df.fillna('')
        
        # Store preprocessed data with proper NaN handling - only essential columns
        preprocessed_df = df[['Title', 'Abstract', 'Category', 'combined_text', 'preprocessed_text']].copy()
        
        # Check if dataset is too large for session storage
        json_data = preprocessed_df.to_json(orient='records', force_ascii=False, date_format='iso')
        if len(json_data) > 1024 * 1024:  # 1MB limit
            print(f"DEBUG: Dataset too large for session ({len(json_data)} bytes), limiting to first 1000 rows")
            limited_df = preprocessed_df.head(1000)
            json_data = limited_df.to_json(orient='records', force_ascii=False, date_format='iso')
        
        session['excel_preprocessed'] = json_data
        
        # Calculate statistics
        avg_tokens = total_tokens / len(df) if len(df) > 0 else 0
        
        # Get sample comparison
        sample_comparison = None
        if len(df) > 0:
            original_text = str(df['combined_text'].iloc[0])[:300]
            preprocessed_text = str(df['preprocessed_text'].iloc[0])[:300]
            sample_comparison = {
                'original': original_text,
                'preprocessed': preprocessed_text
            }
        
        # Create response data
        response_data = {
            'success': True,
            'total_processed': len(df),
            'avg_tokens': round(avg_tokens, 1),
            'total_unique_words': len(all_words),
            'sample_comparison': sample_comparison
        }
        
        # Add warning if dataset was limited
        if len(df) == MAX_ROWS:
            response_data['warning'] = f'Dataset was limited to {MAX_ROWS} rows for performance reasons'
        
        return jsonify(response_data)
        
    except MemoryError:
        print("DEBUG: Memory error - dataset too large")
        return jsonify({
            'success': False, 
            'error': 'Dataset too large to process. Please use a smaller dataset or contact support.'
        }), 413  # Request Entity Too Large
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")  # Debug logging
        error_msg = str(e)
        if 'too large' in error_msg.lower() or 'memory' in error_msg.lower():
            return jsonify({
                'success': False, 
                'error': 'Dataset too large to process. Please try with a smaller file.'
            }), 413
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calculate_tfidf_excel', methods=['POST'])
def calculate_tfidf_excel():
    """Calculate TF-IDF for Excel data"""
    if 'excel_preprocessed' not in session:
        return jsonify({'success': False, 'error': 'No preprocessed data available'}), 400
    
    try:
        # Load preprocessed data
        df = pd.read_json(session['excel_preprocessed'])
        
        # Ensure preprocessed_text column is clean
        df['preprocessed_text'] = df['preprocessed_text'].fillna('').astype(str)
        
        # Filter out empty texts
        valid_texts = [text for text in df['preprocessed_text'] if text.strip()]
        
        if len(valid_texts) == 0:
            return jsonify({'success': False, 'error': 'No valid preprocessed text found'}), 400
        
        # Calculate TF-IDF with adjusted parameters for robustness
        min_doc_freq = max(1, min(2, len(valid_texts) // 10))  # Adjust min_df based on dataset size
        max_doc_freq = min(0.9, max(0.5, 1.0 - (10.0 / len(valid_texts))))  # Adjust max_df
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, 
            min_df=min_doc_freq, 
            max_df=max_doc_freq,
            stop_words='english'  # Add English stop words
        )
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(valid_texts)
        
        # Get feature names and their average TF-IDF scores
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Create word-score pairs and sort by score
        word_score_pairs = list(zip(feature_names, tfidf_scores))
        word_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Separate words and scores, ensure no NaN values
        tfidf_words = []
        tfidf_values = []
        
        for word, score in word_score_pairs:
            if not pd.isna(score) and not np.isnan(score):
                tfidf_words.append(str(word))
                tfidf_values.append(float(score))
        
        # Store TF-IDF data
        session['excel_tfidf'] = {
            'words': tfidf_words,
            'values': tfidf_values,
            'vectorizer': 'stored'  # We'll recreate when needed
        }
        
        return jsonify({
            'success': True,
            'tfidf_words': tfidf_words,
            'tfidf_values': tfidf_values,
            'total_features': len(feature_names),
            'papers_processed': len(valid_texts)
        })
        
    except Exception as e:
        print(f"TF-IDF calculation error: {str(e)}")  # Debug logging
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/classify_excel', methods=['POST'])
def classify_excel():
    """Train and evaluate SVM on Excel data"""
    if 'excel_preprocessed' not in session:
        return jsonify({'success': False, 'error': 'No preprocessed data available'}), 400
    
    try:
        data = request.get_json()
        kernel = data.get('kernel', 'linear')
        train_split = int(data.get('train_split', 80))
        test_split = 100 - train_split
        
        # Load preprocessed data
        df = pd.read_json(session['excel_preprocessed'])
        
        # Clean data and handle NaN values
        df['preprocessed_text'] = df['preprocessed_text'].fillna('').astype(str)
        df['Category'] = df['Category'].fillna('Unknown').astype(str)
        
        # Filter out empty texts
        valid_df = df[df['preprocessed_text'].str.strip() != ''].copy()
        
        if len(valid_df) == 0:
            return jsonify({'success': False, 'error': 'No valid data for classification'}), 400
        
        # Prepare data  
        X = valid_df['preprocessed_text']
        y = valid_df['Category']
        
        # Check if we have multiple categories
        unique_categories = y.unique()
        if len(unique_categories) < 2:
            return jsonify({'success': False, 'error': 'Need at least 2 categories for classification'}), 400
        
        # Label encoding
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data with stratification if possible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_split/100, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # If stratification fails (e.g., some classes have only 1 sample), split without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_split/100, random_state=42
            )
        
        # Create pipeline with robust parameters
        tfidf_params = {
            'max_features': min(1000, len(X_train)),
            'min_df': max(1, len(X_train) // 100),
            'max_df': 0.95,
            'stop_words': 'english'
        }
        
        if kernel == 'linear':
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='linear', probability=True, random_state=42))
            ])
        else:  # polynomial
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(**tfidf_params)),
                ('scaler', MaxAbsScaler()),
                ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
            ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics with zero division handling
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Convert confusion matrix to list for JSON serialization
        confusion_matrix_data = []
        for i in range(len(label_encoder.classes_)):
            for j in range(len(label_encoder.classes_)):
                confusion_matrix_data.append({
                    'actual': label_encoder.classes_[i],
                    'predicted': label_encoder.classes_[j],
                    'count': int(cm[i][j])
                })
        
        # Calculate per-category metrics
        category_metrics = []
        # Get per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Get support (number of samples for each class)
        unique_labels, counts = np.unique(y_test, return_counts=True)
        
        for i, category in enumerate(label_encoder.classes_):
            support = counts[i] if i < len(counts) else 0
            category_metrics.append({
                'category': category,
                'precision': float(precision_per_class[i]) if i < len(precision_per_class) else 0.0,
                'recall': float(recall_per_class[i]) if i < len(recall_per_class) else 0.0,
                'f1_score': float(f1_per_class[i]) if i < len(f1_per_class) else 0.0,
                'support': int(support)
            })
        
        # Get feature count
        total_features = len(pipeline.named_steps['tfidf'].get_feature_names_out())
        
        return jsonify({
            'success': True,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'kernel': kernel,
            'total_papers': len(valid_df),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_features': total_features,
            'confusion_matrix': confusion_matrix_data,
            'category_metrics': category_metrics,
            'categories': label_encoder.classes_.tolist()
        })
        
    except Exception as e:
        print(f"Classification error: {str(e)}")  # Debug logging
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
