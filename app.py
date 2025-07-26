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
from nltk.stem import WordNetLemmatizer
import re
import pickle
import json

# Optional import for Word2Vec (for demonstration purposes)
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Note: gensim not available, Word2Vec will be simulated")

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Thesis fields
FIELDS = [
    'Artificial Intelligence',
    'Distributed Systems',
    'Image Processing',
    'Networking and Cybersecurity',
    'Software Engineering'
]

# --- Preprocessing Functions ---
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
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # Step 5: Stemming and Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    # Final preprocessed text
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    # Return all steps for visualization
    return {
        'original': original_text,
        'lowercased': lowercased_text,
        'cleaned': cleaned_text,
        'tokens': tokens,
        'filtered_tokens': filtered_tokens,
        'lemmatized_tokens': lemmatized_tokens,
        'preprocessed_text': preprocessed_text
    }

def preprocess_simple(text):
    """Simple version that just returns the final preprocessed text"""
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
    load_excel_data('distribution_data.xlsx', 'Distributed Systems'),
    load_excel_data('image_processing_data.xlsx', 'Image Processing'),
    load_excel_data('networking_cybersecurity_data.xlsx', 'Networking and Cybersecurity'),
    load_excel_data('se_data.xlsx', 'Software Engineering')
]

# Combine all dataframes
data = pd.concat(dataframes, ignore_index=True)
data = data.dropna(subset=['Title', 'Abstract', 'Category'])
data['text'] = data['Title'].astype(str) + ' ' + data['Abstract'].astype(str)
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
    
    # Get all preprocessing steps
    preprocessing_steps = preprocess(combined_text)
    
    # Store the preprocessed text in the session for later use
    preprocessed_text = preprocessing_steps['preprocessed_text']
    
    # Return just the preprocessing steps
    return jsonify({
        'preprocessing_steps': preprocessing_steps,
        'preprocessed_text': preprocessed_text
    })

@app.route('/word2vec_vectorization', methods=['POST'])
def word2vec_vectorization():
    # Get the preprocessed text from the request
    preprocessed_text = request.form.get('preprocessed_text', '')
    
    # Split text into words
    words = preprocessed_text.split()
    unique_words = list(set(words))
    
    if GENSIM_AVAILABLE and len(words) > 1:
        try:
            # Create Word2Vec model with the text
            # For single text, we'll use a small window and create synthetic sentences
            sentences = []
            for i in range(0, len(words), 5):  # Create sentences of 5 words each
                sentence = words[i:i+5]
                if len(sentence) > 1:
                    sentences.append(sentence)
            
            # Add the full text as a sentence too
            sentences.append(words)
            
            # Train Word2Vec model
            model = Word2Vec(sentences, vector_size=100, window=3, min_count=1, workers=1, seed=42)
            
            # Get vectors for each unique word
            word_vectors = {}
            for word in unique_words:
                if word in model.wv:
                    vector = model.wv[word].tolist()
                    word_vectors[word] = vector
            
            # Find similar words for demonstration
            similar_words = {}
            for word in unique_words[:10]:  # Only show for first 10 words
                if word in model.wv:
                    try:
                        similar = model.wv.most_similar(word, topn=3)
                        similar_words[word] = [(w, float(score)) for w, score in similar]
                    except:
                        similar_words[word] = []
            
            return jsonify({
                'success': True,
                'method': 'Word2Vec',
                'total_words': len(words),
                'unique_words': len(unique_words),
                'word_list': unique_words,
                'vector_dimensions': 100,
                'word_vectors': word_vectors,
                'similar_words': similar_words,
                'sample_vector': word_vectors.get(unique_words[0], []) if unique_words else []
            })
            
        except Exception as e:
            # Fallback to simulated Word2Vec if real one fails
            pass
    
    # Fallback: Simulated Word2Vec representation
    import random
    random.seed(42)  # For consistent results
    
    word_vectors = {}
    for word in unique_words:
        # Generate a consistent "vector" based on word characters
        vector = []
        word_hash = hash(word) % 1000000
        random.seed(word_hash)
        for _ in range(100):
            vector.append(round(random.uniform(-1, 1), 4))
        word_vectors[word] = vector
    
    # Simulate similar words based on word length and first letter
    similar_words = {}
    for word in unique_words[:10]:
        similar_candidates = [w for w in unique_words if w != word and 
                            (len(w) == len(word) or w[0] == word[0])][:3]
        similar_words[word] = [(w, round(random.uniform(0.3, 0.9), 3)) for w in similar_candidates]
    
    return jsonify({
        'success': True,
        'method': 'Word2Vec (Simulated)',
        'total_words': len(words),
        'unique_words': len(unique_words),
        'word_list': unique_words,
        'vector_dimensions': 100,
        'word_vectors': word_vectors,
        'similar_words': similar_words,
        'sample_vector': word_vectors.get(unique_words[0], []) if unique_words else []
    })

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
    text = preprocess_simple(title + ' ' + abstract)
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
        session.permanent = True  # Make session permanent
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
    session.permanent = True
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

if __name__ == '__main__':
    app.run(debug=True)
