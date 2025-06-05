# Paper Field Categorization Web App

This Flask web app allows you to categorize academic papers into 5 fields by entering the title and abstract. It uses NLP preprocessing, TF-IDF, and an SVM classifier.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download NLTK data (only needed first time):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open your browser at `http://127.0.0.1:5000`

## Features
- Enter title and abstract
- Get instant field prediction (5 fields)
- Modern, clean UI

## Project Structure
- `app.py`: Flask backend and ML logic
- `templates/index.html`: Frontend UI
- `static/style.css`: CSS styling
