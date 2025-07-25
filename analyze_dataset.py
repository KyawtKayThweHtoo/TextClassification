import pandas as pd
import os
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_excel_file(file_path, category_name):
    """Analyze an Excel file and extract key information"""
    try:
        df = pd.read_excel(file_path)
        print(f"\n=== {category_name} Dataset Analysis ===")
        print(f"File: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first few rows structure
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        # If there's a text/content column, analyze terminology
        text_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['text', 'content', 'abstract', 'title', 'body'])]
        
        if text_columns:
            print(f"\nText columns found: {text_columns}")
            text_col = text_columns[0]  # Use first text column
            
            # Combine all text for TF-IDF analysis
            all_text = ' '.join(df[text_col].dropna().astype(str))
            
            # Extract key terms using simple word frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
            word_freq = Counter(words)
            
            print(f"\nTop 20 most frequent terms in {category_name}:")
            for word, count in word_freq.most_common(20):
                print(f"  {word}: {count}")
        
        return df
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    data_dir = "/Users/toe/Htoo/TextClassification/data"
    
    datasets = {
        "AI": "ai_data.xlsx",
        "Image Processing": "image_processing_data.xlsx", 
        "Distributed Systems": "distribution_data.xlsx",
        "Networking & Cybersecurity": "networking_cybersecurity_data.xlsx",
        "Software Engineering": "se_data.xlsx"
    }
    
    # Check if pandas and openpyxl are available
    try:
        import openpyxl
    except ImportError:
        print("openpyxl not found. Installing...")
        os.system("pip install openpyxl")
    
    for category, filename in datasets.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            analyze_excel_file(file_path, category)
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main()