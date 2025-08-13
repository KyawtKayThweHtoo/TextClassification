import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import tkinter as tk
from tkinter import messagebox, ttk

# --- Helper Functions ---
def truncate_to_words(text, max_words=500):
    """Truncate text to a maximum number of words"""
    if not text or pd.isna(text):
        return ''
    words = str(text).split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

# Ensure NLTK data is downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    print("NLTK download failed, but continuing...")

# --- Preprocessing Functions ---
def preprocess_text(text):
    """Preprocess the input text"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    try:
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if w not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        return ' '.join(tokens)
    except:
        # Fallback if NLTK fails
        return text

# --- Load Data ---
def load_data():
    # Thesis fields
    FIELDS = [
        'Artificial Intelligence',
        'Distributed Systems',
        'Image Processing',
        'Networking and Cybersecurity',
        'Software Engineering'
    ]
    
    # Check if preprocessed data exists
    if os.path.exists('preprocessed_papers_data.csv'):
        print("Loading preprocessed data...")
        data = pd.read_csv('preprocessed_papers_data.csv')
        data = data.dropna(subset=['Title', 'Abstract', 'Category'])
        data['text'] = data['Title'].astype(str) + ' ' + data['Abstract'].astype(str)
        data['text'] = data['text'].apply(lambda x: truncate_to_words(x, 500))
        data['text'] = data['text'].apply(preprocess_text)
        return data
    
    # If not, load from Excel files
    print("Loading data from Excel files...")
    data_dir = 'data'
    
    # Load data from each Excel file
    dataframes = []
    excel_files = {
        'ai_data.xlsx': 'Artificial Intelligence',
        'distribution_data.xlsx': 'Distributed Systems',
        'image_processing_data.xlsx': 'Image Processing',
        'networking_cybersecurity_data.xlsx': 'Networking and Cybersecurity',
        'se_data.xlsx': 'Software Engineering'
    }
    
    for filename, category in excel_files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_excel(file_path)
                if 'Title' in df.columns and 'Abstract' in df.columns:
                    df['Category'] = category
                    dataframes.append(df[['Title', 'Abstract', 'Category']])
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    # Combine all dataframes
    if dataframes:
        data = pd.concat(dataframes, ignore_index=True)
        data = data.dropna(subset=['Title', 'Abstract', 'Category'])
        data['text'] = data['Title'].astype(str) + ' ' + data['Abstract'].astype(str)
        data['text'] = data['text'].apply(lambda x: truncate_to_words(x, 500))
        data['text'] = data['text'].apply(preprocess_text)
        return data
    else:
        print("No data found. Please ensure data files exist.")
        return None

class SVMComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVM Model Comparison")
        self.root.geometry("1000x600")
        self.root.configure(bg="#f0f0f0")
        
        # Load data
        self.data = load_data()
        if self.data is None:
            messagebox.showerror("Error", "No data found. Please ensure data files exist.")
            self.root.destroy()
            return
            
        # Label Encoding
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.data['Category'])
        
        # Split for metrics
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data['text'], self.y, test_size=0.2, random_state=42, stratify=self.y)
        
        # --- Pipelines ---
        self.linear_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('scaler', MaxAbsScaler()),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ])
        
        self.poly_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('scaler', MaxAbsScaler()),
            ('svm', SVC(kernel='poly', degree=2, gamma='scale', C=1.0, probability=True, random_state=42))
        ])
        
        # Train models
        print("Training linear SVM model...")
        self.linear_pipeline.fit(self.X_train, self.y_train)
        
        print("Training polynomial SVM model...")
        self.poly_pipeline.fit(self.X_train, self.y_train)
        
        # Calculate metrics
        self.y_pred_linear = self.linear_pipeline.predict(self.X_test)
        self.y_pred_poly = self.poly_pipeline.predict(self.X_test)
        
        self.linear_accuracy = accuracy_score(self.y_test, self.y_pred_linear)
        self.poly_accuracy = accuracy_score(self.y_test, self.y_pred_poly)
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="SVM Model Comparison", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        # Model metrics
        metrics_frame = ttk.LabelFrame(main_frame, text="Model Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, pady=10)
        
        linear_acc_label = ttk.Label(metrics_frame, text=f"Linear SVM Accuracy: {self.linear_accuracy:.4f}")
        linear_acc_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        poly_acc_label = ttk.Label(metrics_frame, text=f"Polynomial SVM Accuracy: {self.poly_accuracy:.4f}")
        poly_acc_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Text Classification", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Title input
        title_label = ttk.Label(input_frame, text="Title:")
        title_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.title_entry = ttk.Entry(input_frame, width=80)
        self.title_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        # Abstract input
        abstract_label = ttk.Label(input_frame, text="Abstract:")
        abstract_label.grid(row=1, column=0, padx=10, pady=5, sticky="nw")
        
        self.abstract_text = tk.Text(input_frame, height=10, width=80, wrap=tk.WORD)
        self.abstract_text.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        
        # Scrollbar for abstract
        abstract_scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=self.abstract_text.yview)
        abstract_scrollbar.grid(row=1, column=2, sticky="ns")
        self.abstract_text.config(yscrollcommand=abstract_scrollbar.set)
        
        # Kernel selection
        kernel_label = ttk.Label(input_frame, text="Kernel:")
        kernel_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        
        self.kernel_var = tk.StringVar(value="linear")
        kernel_frame = ttk.Frame(input_frame)
        kernel_frame.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        
        linear_radio = ttk.Radiobutton(kernel_frame, text="Linear", variable=self.kernel_var, value="linear")
        linear_radio.pack(side=tk.LEFT, padx=10)
        
        poly_radio = ttk.Radiobutton(kernel_frame, text="Polynomial", variable=self.kernel_var, value="poly")
        poly_radio.pack(side=tk.LEFT, padx=10)
        
        # Classify button
        classify_button = ttk.Button(input_frame, text="Classify", command=self.classify_text)
        classify_button.grid(row=3, column=1, padx=10, pady=10, sticky="e")
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding=10)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Results text area
        self.results_text = tk.Text(self.results_frame, height=10, width=80, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)
        input_frame.rowconfigure(1, weight=1)
        
        # Display initial results
        self.display_model_comparison()
        
    def display_model_comparison(self):
        # Display model comparison in the results text area
        self.results_text.delete("1.0", tk.END)
        
        # Get categories
        categories = self.label_encoder.classes_
        
        # Calculate per-class accuracy for linear model
        self.results_text.insert(tk.END, "LINEAR SVM MODEL PERFORMANCE\n", "heading")
        self.results_text.insert(tk.END, "=" * 50 + "\n")
        
        for i, cat in enumerate(categories):
            cat_idx = np.where(self.label_encoder.transform([cat])[0] == self.y_test)[0]
            if len(cat_idx) > 0:
                cat_acc = accuracy_score(
                    self.y_test[cat_idx], 
                    self.y_pred_linear[cat_idx]
                )
                self.results_text.insert(tk.END, f"{cat}: {cat_acc:.4f}\n")
        
        self.results_text.insert(tk.END, f"Overall Accuracy: {self.linear_accuracy:.4f}\n\n")
        
        # Calculate per-class accuracy for polynomial model
        self.results_text.insert(tk.END, "POLYNOMIAL SVM MODEL PERFORMANCE\n", "heading")
        self.results_text.insert(tk.END, "=" * 50 + "\n")
        
        for i, cat in enumerate(categories):
            cat_idx = np.where(self.label_encoder.transform([cat])[0] == self.y_test)[0]
            if len(cat_idx) > 0:
                cat_acc = accuracy_score(
                    self.y_test[cat_idx], 
                    self.y_pred_poly[cat_idx]
                )
                self.results_text.insert(tk.END, f"{cat}: {cat_acc:.4f}\n")
        
        self.results_text.insert(tk.END, f"Overall Accuracy: {self.poly_accuracy:.4f}\n")
        
    def classify_text(self):
        # Get input text
        title = self.title_entry.get()
        abstract = self.abstract_text.get("1.0", tk.END)
        
        if not title and not abstract:
            messagebox.showwarning("Warning", "Please enter a title or abstract.")
            return
        
        # Preprocess text
        combined_text = title + " " + abstract
        combined_text = truncate_to_words(combined_text, 500)
        text = preprocess_text(combined_text)
        
        # Get selected kernel
        kernel = self.kernel_var.get()
        
        # Make prediction
        if kernel == "linear":
            pipeline = self.linear_pipeline
        else:
            pipeline = self.poly_pipeline
            
        pred_code = pipeline.predict([text])[0]
        pred_proba = pipeline.predict_proba([text])[0]
        pred_category = self.label_encoder.inverse_transform([pred_code])[0]
        confidence = pred_proba[pred_code] * 100
        
        # Show result in message box
        result_message = f"Classification Result:\n\n" \
                         f"Category: {pred_category}\n" \
                         f"Confidence: {confidence:.2f}%\n\n" \
                         f"Model: SVM with {kernel} kernel"
        
        messagebox.showinfo("Classification Result", result_message)
        
        # Show all probabilities
        categories = self.label_encoder.classes_
        proba_message = "Probability Distribution:\n\n"
        
        for i, category in enumerate(categories):
            proba_message += f"{category}: {pred_proba[i] * 100:.2f}%\n"
            
        messagebox.showinfo("Probability Distribution", proba_message)

if __name__ == "__main__":
    root = tk.Tk()
    app = SVMComparisonApp(root)
    root.mainloop()
