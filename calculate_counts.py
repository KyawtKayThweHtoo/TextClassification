import pandas as pd

# Thesis fields as defined in app.py
FIELDS = [
    'Artificial Intelligence',
    'Distribution Data',
    'Image Preprocessing',
    'Networking and Cybersecurity',
    'Software Engineering'
]

try:
    # Load data similar to app.py
    data_df = pd.read_csv('preprocessed_papers_data.csv')
    
    # Show total count BEFORE filtering
    initial_total_papers = len(data_df)
    print(f"Actual total rows in preprocessed_papers_data.csv (before any filtering): {initial_total_papers}")

    # Match app.py's data cleaning for count consistency with the main app
    data_df = data_df.dropna(subset=['Title', 'Abstract', 'Category'])

    total_papers_after_filtering = len(data_df)
    print(f"Total papers from preprocessed_papers_data.csv after filtering (matching app.py logic): {total_papers_after_filtering}")

    # Use the filtered dataframe for category counts
    category_counts = data_df['Category'].value_counts()

    category_counts = data_df['Category'].value_counts()

    print("\nPapers per specified field (from preprocessed_papers_data.csv after filtering):")
    for field in FIELDS:
        count = category_counts.get(field, 0)
        print(f"- {field}: {count}")

    print("\nAll categories found in preprocessed_papers_data.csv and their counts (after filtering):")
    if not category_counts.empty:
        for category, count in category_counts.items():
            print(f"- {category}: {count}")
    else:
        print("No categories found or the 'Category' column is empty after filtering.")

except FileNotFoundError:
    print("Error: 'preprocessed_papers_data.csv' not found in the current directory.")
    print("Please ensure the CSV file is in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
