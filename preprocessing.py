import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import nltk
nltk.download('punkt_tab')



# Step 1: Load all Excel files
data_folder = 'data'
excel_files = [
    'ai_data.xlsx',
    'distribution_data.xlsx',
    'image_processing_data.xlsx',
    'networking_cybersecurity_data.xlsx',
    'se_data.xlsx'
]

all_data = []

for file in excel_files:
    path = os.path.join(data_folder, file)
    df = pd.read_excel(path)
    
    # Only keep necessary columns
    df = df[['Title', 'Abstract', 'Category']]
    
    # Fill NaN with empty strings
    df.fillna('', inplace=True)

    # Combine title and abstract into a single text field
    df['text'] = df['Title'] + ' ' + df['Abstract']
    
    all_data.append(df[['text', 'Category']])

# Combine all into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(combined_df['text'])
y = combined_df['Category']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train SVM Classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
